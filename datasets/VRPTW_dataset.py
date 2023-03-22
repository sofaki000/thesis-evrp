from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os
import time
import numpy as np

from rl_for_solving_the_vrp.src import config

from or_tools_comparisons.time_window_constraints_vrp.time_windows_utilities import get_start_and_end_times, \
    get_batch_for_time_intervals

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
velocity = 10

add_penalty_for_extra_time = True
def reward_fn_vrptw(static, tour_indices,time_spent_at_each_route):
    # TODO: add penalty when not visiting city/ visiting not in schedule
    # TODO 2: add reward for less time waiting aka less time spent in routes
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0 # i was ensuring this earlier!! don't know which place is better
    # to ensure this
    # start = static.data[:, :, 0].unsqueeze(1)
    # y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))

    if add_penalty_for_extra_time:
        return tour_len.sum(1) - time_spent_at_each_route

    return tour_len.sum(1)


class VRPTW_data_simple(Dataset):
    def __init__(self, train_size, num_nodes):
        super().__init__()
        seed = 520
        self.size = train_size

        torch.manual_seed(seed)
        self.velocity = velocity
        customers = torch.rand(train_size, 2, num_nodes, device=device)

        # TODO: create realistic windows + draw gannt diagram for them
        start_times, end_times = get_start_and_end_times()
        # earliest_window = torch.rand(train_size, 1, num_nodes)
        # latest_window = earliest_window + 10
        #time_windows = torch.cat((earliest_window, latest_window), dim=1)

        # earliest_window = torch.tensor(start_times)
        # earliest_window = earliest_window.repeat(train_size).reshape(train_size,num_nodes)
        # latest_window = torch.tensor(end_times)
        # latest_window = latest_window.repeat(train_size).reshape(train_size,num_nodes)
        # time_windows = torch.cat((earliest_window.unsqueeze(1), latest_window.unsqueeze(1)), dim=1)

        time_windows = get_batch_for_time_intervals(train_size)

        self.static = torch.cat((customers, time_windows), dim=1) # (train_size, 4 , num_nodes )

        # we keep the current time at dynamic features
        # we start from time=0.
        self.dynamic = torch.zeros(train_size, 1, num_nodes)
        seq_len = self.static.size(2)
        self.distances = torch.zeros(train_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            self.distances[:, i] = self.cal_dis(self.static[:, 0, :], self.static[:, 1, :], self.static[:, 0, i:i+1], self.static[:, 1, i:i+1])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # location = self.static[:, 0:2, :]
        # time_windows = self.static[:, 2:, :]
        # return location[idx], time_windows[idx], self.distances[idx], self.dynamic[idx]
        location = self.static[:, 0:2, :]
        time_windows = self.static[:, 2:, :]
        return  self.static[idx], self.dynamic[idx], self.distances[idx]

    def cal_dis(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # degrees to radians
        lon1, lat1, lon2, lat2 = map(lambda x: x / 180 * np.pi, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.pow(torch.sin(dlat / 2), 2) + torch.cos(lat1) * torch.cos(lat2) * torch.pow(torch.sin(dlon / 2), 2)
        c = 2 * 4182.44949 * torch.asin(torch.sqrt(a))  # miles
        # c = 2 * atan2(sqrt(a), sqrt(1 - a)) * 4182.44949
        return c


def update_mask_vrptw(static, dynamic, mask, chosen_indexes):
        '''
        The masked cities are the ones we have visited before and the ones
        we cannot visit again due to time constraints
        '''

        # Theloume na kanoume mask ta elements pou den mporei na paei logo elipsh xronou
        time_windows = static[:, 2:, :]

        min_time_to_visit_each_location = time_windows[:,0]
        max_time_to_visit_each_location = time_windows[:,1]

        current_time  = dynamic.clone()
        # only the first column of each row (each row has the same time)
        current_time = current_time[:, :, 0]

        # TODO 1: If no customer is open yet, introduce waiting time
        # and passing time as car goes to customer
        # TODO 2: add multiple vehicles
        # check if current time is within time_window
        after_start_date = (current_time > min_time_to_visit_each_location)
        before_end_date = (current_time< max_time_to_visit_each_location)

        can_visit_city_time_constraints = np.logical_and(after_start_date.numpy(), before_end_date.numpy())


        # if dynamic.numpy().sum() ==0:
        #     # we are on the first iteration, we masked already
        #     return mask
        # put 0 to all chosen_indexes so we can't revisit them
        mask.scatter_(1, chosen_indexes.unsqueeze(1), 0)

        # 1 we can visit, 0 we cant visit
        mask2 = torch.tensor(can_visit_city_time_constraints).type(torch.uint8)

        hasnt_visited_before = mask.numpy() == 1

        if mask2.byte().any():
            # at this time no customer is accessible
            # we go to a customer and wait
            #print("We have to wait!")
            final_mask =  hasnt_visited_before
        else:
            isnt_within_time_limits = mask2.numpy()==1
            final_mask = hasnt_visited_before | isnt_within_time_limits

        return torch.tensor(final_mask).type(torch.uint8)

def update_dynamic_state_vrptw(static, dynamic, idx, old_idx, distances):
    """
        :param old_idx: (batch*beam, 1)
        :param idx: ditto
        :param mask: (batch*beam, seq_len)
        :param dynamic: (batch*beam, dynamic_features, seq_len)
        :param distances: (batch*beam, seq_len, seq_len)
        :param dis_by_afs: (batch*beam, seq_len)
        :param capacity, velocity, cons_rate, t_limit, num_afs: scalar
        :return: updated dynamic
    """
    batch_size, num_features, num_nodes = static.size()
    # briskoume to distance pou ekane apo to palio index sto kainourgio index
    dis = distances[torch.arange(distances.size(0)), old_idx.long() , idx.long()].unsqueeze(1)

    time = dynamic.clone()
    time_windows = static[:, 2:, :]

    earliest_time = time_windows[:,0]
    arr = np.array(earliest_time)

    times_each_chosen_city_can_be_visited = arr[np.arange(arr.shape[0]), idx]
    times_for_each_route = time.squeeze(1)[:,0]

    time_we_arrive_at_each_city = times_for_each_route + (dis / velocity).squeeze(1)

    arrived_earlier_than_possible = time_we_arrive_at_each_city.numpy() < times_each_chosen_city_can_be_visited

    time_after_waiting = time_we_arrive_at_each_city.clone()
    # se kathe city pou prepei na perimenoume, prosthetoume thn diafora
    time_we_wait = times_each_chosen_city_can_be_visited[arrived_earlier_than_possible]- time_after_waiting[arrived_earlier_than_possible].numpy()
    time_after_waiting[arrived_earlier_than_possible] += time_we_wait

    # we reshape to [batch_size, num_nodes].
    final_time = time_after_waiting.reshape(batch_size,1).expand(batch_size,num_nodes)

    # we return the new dynamic elements
    dynamic = torch.tensor(final_time)

    return dynamic.unsqueeze(1)


class VRPTW_data(Dataset):
    def __init__(self, train_size, num_nodes,  t_limit,  velocity=40):
        super().__init__()
        seed = 520
        self.size = train_size
        self.velocity = velocity
        self.cons_rate = config.cons_rate

        torch.manual_seed(seed)

        customers = torch.rand(train_size, 2, num_nodes, device=device)
        customers[:, 0, :] = customers[:, 0, :] * 4 - 79.5
        customers[:, 1, :] = customers[:, 1, :] * 3.5 + 36
        self.static = customers # (train_size, 2, num_nodes )

        self.dynamic = torch.ones(train_size, 1, 1+num_nodes, device=device)   # time duration, capacity, demands
        self.dynamic[:, 0, :] *= t_limit # time duration

        seq_len = self.static.size(2)
        self.distances = torch.zeros(train_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            self.distances[:, i] = self.cal_dis(self.static[:, 0, :], self.static[:, 1, :], self.static[:, 0, i:i+1], self.static[:, 1, i:i+1])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1], self.distances[idx]

    def cal_dis(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # degrees to radians
        lon1, lat1, lon2, lat2 = map(lambda x: x / 180 * np.pi, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.pow(torch.sin(dlat / 2), 2) + torch.cos(lat1) * torch.cos(lat2) * torch.pow(torch.sin(dlon / 2), 2)
        c = 2 * 4182.44949 * torch.asin(torch.sqrt(a))  # miles
        # c = 2 * atan2(sqrt(a), sqrt(1 - a)) * 4182.44949
        return c
    def update_mask_vrptw(self, mask, dynamic, idx=None):
        distances = self.batch_distances

        cons_rate = self.cons_rate
        velocity = self.velocity

        # pernoume ta pragmata ta swsta apo to dynamic
        time = dynamic[:, 0, :].clone()

        # den allazoume kati, apla pairnoume oti xreiazomaste gia na ananewsoume to mask
        dis1 = distances[torch.arange(distances.size(0)), idx ].clone()
        fuel_pd0 = self.cons_rate * dis1
        time_pd0 = dis1 / velocity

        fuel_pd1 = cons_rate * dis1
        time_pd1 = (distances[torch.arange(distances.size(0)), idx ] + distances[:, 0, :]) / velocity
        # time_pd1[:, 1:num_afs + 1] += 0.25
        # time_pd1[:, num_afs + 1:] += 0.5

        # path2: ->Node-> Station-> Depot(choose the station making the total distance shortest)
        #dis2 = distances[:, 1:num_afs + 1, :].gather(1, dis_by_afs[1].unsqueeze(1)).squeeze(1)
        #dis2[:, 0] = 0
        #dis2 += distances[torch.arange(distances.size(0)), idx]
        #fuel_pd2 = cons_rate * dis2
        time_pd2 = (distances[torch.arange(distances.size(0)), idx ] + dis_by_afs[0]) / velocity
        #time_pd2[:, 1:num_afs + 1] += 0.25
        #time_pd2[:, num_afs + 1:] += 0.5

        # path3: ->Node-> Station-> Depot(choose the closest station to the node), ignore this path temporarily
        # the next node should be able to return to depot with at least one way; otherwise, mask it

        # afs = (idx.gt(0) & idx.le(num_afs))
        # fs = idx.le(num_afs)
        # customer = idx.gt(num_afs)

        mask.scatter_(1, idx.unsqueeze(1), 0)
        # # forbid passing by afs if leaving depot, allow if returning to depot; forbid from afs to afs, not necessary but convenient
        # mask[fs, 1:num_afs + 1] = 0  # float('-inf')
        # mask[afs, 0] = 0
        # mask[customer, :num_afs + 1] = 0
        # mask[fs] = torch.where(demands[fs] > 0, torch.zeros(mask[fs].size(), device=device), mask[fs])
        #
        # mask[~((fuel >= fuel_pd1) & (time >= time_pd1) | (fuel >= fuel_pd2) & (time >= time_pd2))] = 0  # float('-inf')
        #
        # mask[(fuel < fuel_pd0) | (time < time_pd0)] = 0  # float('-inf')
        #
        # all_masked = mask[:, num_afs + 1:].eq(0).sum(1).le(0)
        # mask[all_masked, 0] = 1  # unmask the depot if all nodes are masked

        return mask
    def update_dynamic_state_vrptw(self, dynamic, idx, old_idx, distances=None):
            """
            :param old_idx: (batch*beam, 1)
            :param idx: ditto
            :param mask: (batch*beam, seq_len)
            :param dynamic: (batch*beam, dynamic_features, seq_len)
            :param distances: (batch*beam, seq_len, seq_len)
            :param dis_by_afs: (batch*beam, seq_len)
            :param capacity, velocity, cons_rate, t_limit, num_afs: scalar
            :return: updated dynamic
            """
            t_limit = config.t_limit
            cons_rate = self.cons_rate
            velocity= self.velocity
            if distances is None:
                distances = self.batch_distances
            else:
                distances = distances
            dis = distances[torch.arange(distances.size(0)), old_idx.squeeze(1), idx].unsqueeze(1)

            depot = idx.eq(0)
            customer = 0 #idx.gt(num_afs)
            time = dynamic[:, 0, :].clone()
            fuel = dynamic[:, 1, :].clone()
            demands = dynamic[:, 2, :].clone()

            time -= dis / velocity
            time[depot] = t_limit
            #time[afs] -= 0.25
            time[customer] -= 0.5


            dynamic = torch.cat((time.unsqueeze(1), fuel.unsqueeze(1), demands.unsqueeze(1)), dim=1).to(device)

            return dynamic


#
# def update_dynamic_state_vrptw( dynamic, idx, old_idx, distances=None):
#             """
#             :param old_idx: (batch*beam, 1)
#             :param idx: ditto
#             :param mask: (batch*beam, seq_len)
#             :param dynamic: (batch*beam, dynamic_features, seq_len)
#             :param distances: (batch*beam, seq_len, seq_len)
#             :param dis_by_afs: (batch*beam, seq_len)
#             :param capacity, velocity, cons_rate, t_limit, num_afs: scalar
#             :return: updated dynamic
#             """
#             t_limit = config.t_limit
#             velocity = 10
#             dis = distances[torch.arange(distances.size(0)), old_idx.squeeze(1), idx].unsqueeze(1)
#
#             depot = idx.eq(0)
#             customer = 0 #idx.gt(num_afs)
#             time = dynamic[:, 0, :].clone()
#             fuel = dynamic[:, 1, :].clone()
#             demands = dynamic[:, 2, :].clone()
#
#             time -= dis / velocity
#             time[depot] = t_limit
#             #time[afs] -= 0.25
#             time[customer] -= 0.5
#
#             #fuel -= cons_rate * dis
#             #fuel[fs] = capacity
#             #demands.scatter_(1, idx.unsqueeze(1), 0)
#
#             dynamic = torch.cat((time.unsqueeze(1), fuel.unsqueeze(1), demands.unsqueeze(1)), dim=1).to(device)
#
#             return dynamic
#
# def update_mask_vrptw( mask, dynamic,distances,  idx=None):
#
#         cons_rate = 0.5 #self.cons_rate
#         velocity = 10 #self.velocity
#
#         # pernoume ta pragmata ta swsta apo to dynamic
#         time = dynamic[:, 0, :].clone()
#
#         # den allazoume kati, apla pairnoume oti xreiazomaste gia na ananewsoume to mask
#         dis1 = distances[torch.arange(distances.size(0)), idx ].clone()
#         #fuel_pd0 = self.cons_rate * dis1
#         time_pd0 = dis1 / velocity
#
#         fuel_pd1 = cons_rate * dis1
#         time_pd1 = (distances[torch.arange(distances.size(0)), idx ] + distances[:, 0, :]) / velocity
#         # time_pd1[:, 1:num_afs + 1] += 0.25
#         # time_pd1[:, num_afs + 1:] += 0.5
#
#         # path2: ->Node-> Station-> Depot(choose the station making the total distance shortest)
#         #dis2 = distances[:, 1:num_afs + 1, :].gather(1, dis_by_afs[1].unsqueeze(1)).squeeze(1)
#         #dis2[:, 0] = 0
#         #dis2 += distances[torch.arange(distances.size(0)), idx]
#         #fuel_pd2 = cons_rate * dis2
#         time_pd2 = (distances[torch.arange(distances.size(0)), idx ] + dis_by_afs[0]) / velocity
#         #time_pd2[:, 1:num_afs + 1] += 0.25
#         #time_pd2[:, num_afs + 1:] += 0.5
#
#         # path3: ->Node-> Station-> Depot(choose the closest station to the node), ignore this path temporarily
#         # the next node should be able to return to depot with at least one way; otherwise, mask it
#
#         # afs = (idx.gt(0) & idx.le(num_afs))
#         # fs = idx.le(num_afs)
#         # customer = idx.gt(num_afs)
#
#         mask.scatter_(1, idx.unsqueeze(1), 0)
#         # # forbid passing by afs if leaving depot, allow if returning to depot; forbid from afs to afs, not necessary but convenient
#         # mask[fs, 1:num_afs + 1] = 0  # float('-inf')
#         # mask[afs, 0] = 0
#         # mask[customer, :num_afs + 1] = 0
#         # mask[fs] = torch.where(demands[fs] > 0, torch.zeros(mask[fs].size(), device=device), mask[fs])
#         #
#         # mask[~((fuel >= fuel_pd1) & (time >= time_pd1) | (fuel >= fuel_pd2) & (time >= time_pd2))] = 0  # float('-inf')
#         #
#         # mask[(fuel < fuel_pd0) | (time < time_pd0)] = 0  # float('-inf')
#         #
#         # all_masked = mask[:, num_afs + 1:].eq(0).sum(1).le(0)
#         # mask[all_masked, 0] = 1  # unmask the depot if all nodes are masked
#
#         return mask
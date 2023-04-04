from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GVRPDataset(Dataset):
    def __init__(self, train_size, num_nodes, t_limit, capacity, num_afs=3, data_dir=None, seed=520):
        super().__init__()
        self.size = train_size
        if data_dir:
            depot = set()
            afs = set()
            self.static = torch.zeros(10, 2, 1+num_afs+num_nodes, device=device)

            for i in range(10):
                filename = f'20c3sU{i + 1}.txt'
                x = []
                y = []
                with open(os.path.join(data_dir, filename), 'r') as f:
                    lines = f.readlines()[2:26]
                    depot.add((float(lines[0].split()[2]), float(lines[0].split()[3])))
                    for k in range(1, num_afs+1):
                        afs.add((float(lines[k].split()[2]), float(lines[k].split()[3])))
                    for line in lines[num_afs+1:]:
                        line = line.strip()
                        if not line.startswith('C'):
                            print(line)
                            raise ValueError(f'the format of {filename} is not consistent with the others')
                        line = line.split()
                        x.append(float(line[2]))
                        y.append(float(line[3]))
                self.static[i, 0, num_afs+1:] = torch.tensor(x)
                self.static[i, 1, num_afs+1:] = torch.tensor(y)

            assert len(depot) == 1 and len(afs) == num_afs
            self.static[:, :, 0] = torch.tensor(list(depot)).unsqueeze(0)
            self.static[:, :, 1:num_afs+1] = torch.tensor(sorted(list(afs), reverse=True)).transpose(1, 0).unsqueeze(0)
        else:
            torch.manual_seed(seed)
            # # left bottom: (-79.5, 36); top right: (-75.5, 39.5)
            afs = torch.tensor([[-76.338677, -77.08760885, -79.156076], [36.796046, 39.45787498, 37.383343]])
            afs = torch.cat((torch.tensor([-77.49439265, 37.60851245]).unsqueeze(1), afs), dim=1).to(device)  # add depot
            # afs[0] = (afs[0] + 79.5)/4
            # afs[1] = (afs[1] - 36)/3.5
            customers = torch.rand(train_size, 2, num_nodes, device=device)
            customers[:, 0, :] = customers[:, 0, :] * 4 - 79.5
            customers[:, 1, :] = customers[:, 1, :] * 3.5 + 36
            self.static = torch.cat((afs.unsqueeze(0).repeat(train_size, 1, 1), customers), dim=2).to(device)  # (train_size, 2, num_nodes+4)

        self.dynamic = torch.ones(train_size, 3, 1+num_afs+num_nodes, device=device)   # time duration, capacity, demands
        self.dynamic[:, 0, :] *= t_limit
        self.dynamic[:, 1, :] *= capacity
        self.dynamic[:, 2, :num_afs+1] = 0
        # self.dynamic[:, 1, :self.num_afs+1] = 0

        seq_len = self.static.size(2)
        self.distances = torch.zeros(train_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            self.distances[:, i] = cal_dis(self.static[:, 0, :], self.static[:, 1, :], self.static[:, 0, i:i+1], self.static[:, 1, i:i+1])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.static[idx], self.dynamic[idx], self.distances[idx]    # dynamic: None

def cal_dis(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # degrees to radians
    lon1, lat1, lon2, lat2 = map(lambda x: x/180*np.pi, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.pow(torch.sin(dlat / 2), 2) + torch.cos(lat1) * torch.cos(lat2) * torch.pow(torch.sin(dlon / 2), 2)
    c = 2 * 4182.44949 * torch.asin(torch.sqrt(a))    # miles
    # c = 2 * atan2(sqrt(a), sqrt(1 - a)) * 4182.44949
    return c

def reward_func(tours, static, distances, beam_width=1):
    """
    :param tours: LongTensor, (batch*beam, seq_len)
    :param static: (batch, 2, num_nodes)
    :param distances: (batch, num_nodes, num_nodes)
    :param beam_width: set beam_width=1 when training
    :return: reward: Euclidean distance between each consecutive point， (batch)
    :return: locs: (batch, 2, seq_len)
    """
    bb_size, seq_len = tours.size()
    batch_size = static.size(0)
    depot = torch.zeros(bb_size, 1, dtype=torch.long, device=device)
    tours = torch.cat((depot, tours, depot), dim=1)         # start from depot, end at depot(although some have ended at depot)
    id0 = torch.arange(bb_size).unsqueeze(1).repeat(1, seq_len+1)
    reward = distances.repeat(beam_width, 1, 1)[id0, tours[:, :-1], tours[:, 1:]].sum(1)    # (batch*beam)
    # (batch*beam) -> (batch), choose the best reward
    reward, id_best = torch.cat(torch.chunk(reward.unsqueeze(1), beam_width, dim=0), dim=1).min(1)  # (batch)
    bb_idx = torch.arange(batch_size, device=device) + id_best * batch_size
    tours = tours[bb_idx]
    # print(tours)
    tours = tours.unsqueeze(1).repeat(1, static.size(1), 1)
    locs = torch.gather(static, dim=2, index=tours)  # (batch, 2, seq_len+)
    return reward, locs


def update_fn(old_idx, idx, mask, dynamic, distances, dis_by_afs, capacity, velocity, cons_rate, t_limit, num_afs):
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
    dis = distances[torch.arange(distances.size(0)), old_idx.squeeze(1), idx.squeeze(1)].unsqueeze(1)
    depot = idx.eq(0).squeeze(1)
    afs = (idx.gt(0) & idx.le(num_afs)).squeeze(1)
    fs = idx.le(num_afs).squeeze(1)
    customer = idx.gt(num_afs).squeeze(1)    # TODO: introduce num_afs
    time = dynamic[:, 0, :].clone()
    fuel = dynamic[:, 1, :].clone()
    demands = dynamic[:, 2, :].clone()

    time -= dis/velocity
    time[depot] = t_limit
    time[afs] -= 0.25
    time[customer] -= 0.5

    fuel -= cons_rate * dis
    fuel[fs] = capacity
    demands.scatter_(1, idx, 0)

    dynamic = torch.cat((time.unsqueeze(1), fuel.unsqueeze(1), demands.unsqueeze(1)), dim=1).to(device)

    mask.scatter_(1, idx, float('-inf'))
    # forbid passing by afs if leaving depot, allow if returning to depot; forbid from afs to afs, not necessary but convenient
    mask[fs, 1:num_afs+1] = float('-inf')
    mask[afs, 0] = 0

    mask[customer, :num_afs+1] = 0
    mask[fs] = torch.where(demands[fs] > 0, torch.zeros(mask[fs].size(), device=device), mask[fs])

    # path1: ->Node->Depot
    dis1 = distances[torch.arange(distances.size(0)), idx.squeeze(1)].clone()
    fuel_pd0 = cons_rate * dis1
    time_pd0 = dis1 / velocity
    dis1[:, num_afs+1:] += distances[:, 0, num_afs+1:]
    fuel_pd1 = cons_rate * dis1
    time_pd1 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + distances[:, 0, :]) / velocity
    time_pd1[:, 1:num_afs + 1] += 0.25
    time_pd1[:, num_afs + 1:] += 0.5

    # path2: ->Node-> Station-> Depot(choose the station making the total distance shortest)
    dis2 = distances[:, 1:num_afs+1, :].gather(1, dis_by_afs[1].unsqueeze(1)).squeeze(1)
    dis2[:, 0] = 0
    dis2 += distances[torch.arange(distances.size(0)), idx.squeeze(1)]
    fuel_pd2 = cons_rate * dis2
    time_pd2 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + dis_by_afs[0]) / velocity
    time_pd2[:, 1:num_afs + 1] += 0.25
    time_pd2[:, num_afs + 1:] += 0.5

    # path3: ->Node-> Station-> Depot(choose the closest station to the node), ignore this path temporarily
    # the next node should be able to return to depot with at least one way; otherwise, mask it
    mask[~((fuel >= fuel_pd1) & (time >= time_pd1) | (fuel >= fuel_pd2) & (time >= time_pd2))] = float('-inf')

    mask[(fuel < fuel_pd0) | (time < time_pd0)] = float('-inf')

    all_masked = mask[:, num_afs+1:].eq(0).sum(1).le(0)
    mask[all_masked, 0] = 0  # unmask the depot if all nodes are masked

    return dynamic
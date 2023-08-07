import math

import numpy as np
import torch


def get_tour_distance(number_of_cities, nodes, tour):
    '''
    nodes:(city_t,2), tour:(city_t)
    l(= total distance) = l(0-1) + l(1-2) + l(2-3) + ... + l(18-19) + l(19-0) @20%20->0
    return l:(1)
    '''
    l = 0
    for i in range(number_of_cities):
        l += get_2city_distance(nodes[tour[i]], nodes[tour[(i + 1) % number_of_cities]])
    return l


def stack_l_aka_get_reward(self, inputs, tours):
    '''
    inputs:(batch,city_t,2)
    tours:(batch,city_t)
    return l_batch:(batch)
    '''
    list = [self.get_tour_distance(inputs[i], tours[i]) for i in range(self.batch)]
    l_batch = torch.stack(list, dim=0)
    return l_batch

def get_2city_distance(n1, n2):
    x1, y1, x2, y2 = n1[0], n1[1], n2[0], n2[1]
    if isinstance(n1, torch.Tensor):
        return torch.sqrt((x2 - x1).pow(2) + (y2 - y1).pow(2))
    elif isinstance(n1, (list, np.ndarray)):
        return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    else:
        raise TypeError

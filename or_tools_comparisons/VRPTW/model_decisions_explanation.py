import torch
from or_tools_comparisons.VRPTW.main import get_model_vrptw
from or_tools_comparisons.VRPTW.use_cases.use_cases_utilities import get_data_for_use_case1
from ploting.plot_utilities import show_tour_for_one_solution

## This file is for understanding models decision based on a specific use case scenario
locations, distance_matrix, time_windows = get_data_for_use_case1()


model = get_model_vrptw(use_pretrained_model=True)

num_nodes = locations.shape[1]

static_elements = torch.tensor(torch.cat((torch.tensor(locations), torch.tensor(time_windows))).unsqueeze(0), dtype=torch.float)
dynamic_elements = torch.zeros(1, 1, num_nodes)

vrptw_tour_indices, _, _ = model(static_elements,
                                 dynamic_elements,
                                 torch.tensor(distance_matrix ,dtype=torch.float).unsqueeze(0))


show_tour_for_one_solution(torch.tensor(locations).transpose(1,0) ,
                           distance_matrix,
                           vrptw_tour_indices[0],
                           'use_case_1_solution_model',
                           title="Use case 1, model solution")



####### FOR understanding our solveres we use a use case with VRPTW.

import numpy as np

from or_tools_comparisons.OR_TOOLS.ortools_vrptw_solver import solve_vrp_with_time_windows_with_or_tools, \
    create_data_model_vrptw
from or_tools_comparisons.VRPTW.use_cases.use_cases_utilities import  get_data_for_use_case1
import torch

from plot_utilities import show_tour_for_one_solution


def calculate_total_metric_given_metric_matrix(distance_matrix, route, metric_name):
    # Initialize total distance to zero
    total_distance = 0

    # Loop through the route
    for i in range(len(route)-1):
        # Get the distance between the current node and the next node in the route
        metric = distance_matrix[route[i]][route[i+1]]

        print(f'From {route[i]} to {route[i+1]}:  {metric_name}={metric}')
        # Add the distance to the total distance
        total_distance += metric

    # Return the total metric
    print(f'Total {metric_name} = {total_distance}')
    return total_distance



locations, distance_matrix, time_windows = get_data_for_use_case1()
velocity= 10
print(f'DISTANCE MATRIX: {distance_matrix}')
time_matrix = distance_matrix # / velocity

data_for_or_model = create_data_model_vrptw(time_matrix=torch.tensor(time_matrix), time_windows= torch.tensor(time_windows).transpose(1,0),num_vehicles=1)

or_tools_solution = solve_vrp_with_time_windows_with_or_tools(data_for_or_model)[0]


# for distance
calculate_total_metric_given_metric_matrix(distance_matrix, or_tools_solution, 'distance')

# for time
calculate_total_metric_given_metric_matrix(time_matrix, or_tools_solution, 'time')

# plot solution
locations = torch.tensor(locations).transpose(1,0)

show_tour_for_one_solution(locations , distance_matrix,
                           or_tools_solution,
                           'use_case_1_solution_or_tools',title="Use case 1, or-tools solution")
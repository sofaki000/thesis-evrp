import numpy as np

from or_tools_comparisons.VRPTW.time_windows_utilities import create_gaant_chart_from_times


def write_arrays_to_file(array1, array2, filename):
    with open(filename, 'w') as f:
        for a1, a2 in zip(array1, array2):
            f.write("{},{}\n".format(a1, a2))

def read_arrays_from_file(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=np.int)
    array1 = data[:, 0]
    array2 = data[:, 1]
    return array1, array2


time_windows_contraints_usecase1 = "C:\\Users\\Lenovo\\Desktop\\my_evrp_final\\thesis-evrp\or_tools_comparisons\\VRPTW\\use_cases\\use_case_1_time_windows.csv"
locations_usecase1 = 'C:\\Users\\Lenovo\\Desktop\\my_evrp_final\\thesis-evrp\or_tools_comparisons\\VRPTW\\use_cases\\use_case_1_locations.csv'

time_windows_contraints_usecase2 = "C:\\Users\\Lenovo\\Desktop\\my_evrp_final\\thesis-evrp\or_tools_comparisons\\VRPTW\\use_cases\\use_case_2_time_windows.csv"
locations_usecase2 = 'C:\\Users\\Lenovo\\Desktop\\my_evrp_final\\thesis-evrp\or_tools_comparisons\\VRPTW\\use_cases\\use_case_2_locations.csv'

def compute_distance_matrix(array1, array2):
    n = len(array1)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = array1[i] - array1[j]
            dy = array2[i] - array2[j]
            distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
    return distance_matrix


def get_data_for_use_case1():
    return get_data_for_use_case(locations_usecase1, time_windows_contraints_usecase1,"gaant_user_case_1.png")


def get_data_for_use_case2():
    return get_data_for_use_case(locations_usecase2, time_windows_contraints_usecase2,"gaant_user_case_2.png")

def get_data_for_use_case(file_name_for_locations, file_name_for_time_window_constraints, gaant_name):
    # use case 1 contains time windows and locations
    start_times, end_times = read_arrays_from_file(file_name_for_time_window_constraints)
    xs, ys = read_arrays_from_file(file_name_for_locations)

    distance_matrix = compute_distance_matrix(xs, ys)
    locations = np.vstack((xs, ys))
    time_windows = np.vstack((start_times,end_times))

    # print gaant diagram
    create_gaant_chart_from_times(start_times, end_times, filename=gaant_name)
    return locations, distance_matrix, time_windows


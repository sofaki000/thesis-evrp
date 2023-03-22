"""Vehicles Routing Problem (VRP) with Time Windows."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from or_tools_comparisons.common_utilities import get_routes

time_matrix_default =  [
        [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
        [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
        [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
        [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
        [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
        [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
        [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
        [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
        [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
        [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
        [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
        [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
        [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
        [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
        [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
        [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
        [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0],
    ]

time_windows_default = [
        (0, 5),  # depot
        (7, 12),  # 1
        (10, 15),  # 2
        (16, 18),  # 3
        (10, 13),  # 4
        (0, 5),  # 5
        (5, 10),  # 6
        (0, 4),  # 7
        (5, 10),  # 8
        (0, 3),  # 9
        (10, 16),  # 10
        (10, 15),  # 11
        (0, 5),  # 12
        (5, 10),  # 13
        (7, 8),  # 14
        (10, 15),  # 15
        (11, 15),  # 16
    ]

num_vehicles = 4

def create_data_model_vrptw(time_matrix=time_matrix_default, time_windows= time_windows_default, num_vehicles=num_vehicles):
    """Stores the data for the problem."""
    data = {}
    data['time_matrix'] = time_matrix
    data['time_windows'] =  time_windows
    data['num_vehicles'] =num_vehicles
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    solution.Min(time_var),
                                                    solution.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            solution.Min(time_var))
        print(plan_output)
        total_time += solution.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time))



# In the above function, we first create the routing index manager and the routing model. We then define the distance, time window, and time penalty callbacks using lambda functions. The time penalty callback computes the penalty for violating the time windows. We register the distance and time window callbacks as transit callbacks, and we register the time penalty callback as a unary transit callback. We add a capacity dimension to the routing model to handle vehicle capacities.
#
# We then
def solve_vrp_with_time_windows_with_or_tools_chatgpt_suggestion(data):
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                               data['num_vehicles'],
                                               data['depot'])

        # Create the routing model.
        routing = pywrapcp.RoutingModel(manager)

        # Define the distance callback.
        def distance_callback(from_index, to_index):
            return data['time_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define the time window callback.
        def time_window_callback(node_index):
            return (data['time_windows'][node_index][0], data['time_windows'][node_index][1])

        time_callback_index = routing.RegisterTransitCallback(time_window_callback)

        # Define the time penalty callback.
        def time_penalty_callback(from_index, to_index):
            service_time = 0
            if from_index == 0:
                return 0
            travel_time = distance_callback(from_index, to_index)
            arrival_time = routing.CumulVar(to_index, 'Time')
            time_window = time_window_callback(to_index)
            if arrival_time > time_window[1]:
                delay = arrival_time - time_window[1]
                return delay + travel_time + service_time
            elif arrival_time < time_window[0]:
                delay = time_window[0] - arrival_time
                return delay
            return 0

        routing.SetArcCostEvaluatorOfDimension(transit_callback_index, 'Time')
        penalty_callback_index = routing.RegisterUnaryTransitCallback(time_penalty_callback)
        routing.AddDimensionWithVehicleCapacity(
            penalty_callback_index,
            0,
            [data['vehicle_capacities']] * data['num_vehicles'],
            True,
            'Capacity')

        # Set search parameters.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 10

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Return the solution.
        if solution:
            routes = []
            for vehicle_id in range(data['num_vehicles']):
                route = []
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route.append(node_index)
                    index = solution.Value(routing.NextVar(index))
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                routes.append(route)
            return routes
        else:
            return None


def solve_vrp_with_time_windows_with_or_tools(data):
    """Solve the VRP with time windows."""
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        evaluator_index= transit_callback_index,
        slack_max=1000,  # allow waiting time, slack max specifies the max slack (delay) allowed for each location
        capacity_dim=0, # set to zero bc we dont consider capacity constraints
        fix_start_cumul_to_zero=True, # the cumulative time at the start location is set to zero
        name='Time')
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(int(time_window[0].numpy()), int(time_window[1].numpy()))


    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
           int( data['time_windows'][depot_idx][0].numpy()),
            int(data['time_windows'][depot_idx][1].numpy()))

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    routes = get_routes(solution, routing, manager)
    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)

    return routes


if __name__ == '__main__':
    # Instantiate the data problem.
    data = create_data_model_vrptw()
    solve_vrp_with_time_windows_with_or_tools(data)
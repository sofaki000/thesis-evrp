"""Vehicles Routing Problem (VRP) with Time Windows."""
import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from or_tools_comparisons.common_utilities import get_routes

def create_data_model_vrptw(time_matrix, time_windows, num_vehicles):
    """Stores the data for the problem."""
    data = {}
    data['time_matrix'] = time_matrix
    data['time_windows'] = time_windows
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
        index_outputs = ""
        current_time = ""
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)

            visited_index = manager.IndexToNode(index)

            arrival_time = solution.Value(time_dimension.CumulVar(index ))

            # output cumulative current time dimension
            cumulative_current_time = time_dimension.CumulVar(index)  # total time spent
            # TODO: understand how to get current time of visiting index.
            # Question trying to answer: did we visit index on correct time?
            current_time += f'When visiting index {visited_index}  time is {arrival_time}\n' #, min {solution.Min(time_var)}, max {solution.Max(time_var)} \n'

            index_outputs += f'{visited_index} ->'
            plan_output += f'Node index:{visited_index}'
            plan_output += '{0} Time({1},{2}) -> '.format(visited_index, solution.Min(time_var), solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))



        time_var = time_dimension.CumulVar(index) #total time spent

        print(current_time)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index), solution.Min(time_var), solution.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format( solution.Min(time_var))
        print(plan_output)

        print(f'ROUTE IS: {index_outputs}')
        total_time += solution.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time))




def solve_vrp_with_time_windows_with_or_tools(data):
    """Solve the VRP with time windows."""
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Define the cost function (distance)
    def objective_function_to_minimize(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        travel_time_from_one_node_to_other = data['time_matrix'][from_node][to_node]
        print(f'From node {from_node} to {to_node}: time_travel={travel_time_from_one_node_to_other}')

        return torch.tensor(travel_time_from_one_node_to_other, dtype=int)

    # we set to minimize this function
    transit_callback_index = routing.RegisterTransitCallback(objective_function_to_minimize)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        travel_time_from_one_node_to_other = data['time_matrix'][from_node][to_node]
        print(f'From node {from_node} to {to_node}: time_travel={travel_time_from_one_node_to_other}')

        return torch.tensor(travel_time_from_one_node_to_other,dtype=int)


    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add Time Windows constraint.
    time = 'Time'

    # anexarthta apo to dimension max value briskei thn grhgoroterh lysh. Opote bazoume to dimension_max_value
    # mia megalh posothta
    allowed_waiting_time = 100
    dimension_max_value = 28000 #43200
    routing.AddDimension(transit_callback_index,
                         allowed_waiting_time,
                         dimension_max_value,
                         False,
                         'Time')

    # we get the time dimension. if node doesn't have that dimension, program raised an exception
    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):

        print("ENUMERATING")

        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)

        start_time = int(time_window[0].numpy())
        end_time = int(time_window[1].numpy())
        print(f'For index:{index} range is:[{start_time}, {end_time}] \n')
        # CumulVar represents the cumulative value of a dimension along a route in a VRP
        # it is used to model constraints that involve the accumulation of resources or costs over time
        time_dimension.CumulVar(index).SetRange(start_time, end_time)


    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        start_time =    int( data['time_windows'][depot_idx][0].numpy())
        end_time = int(data['time_windows'][depot_idx][1].numpy())
        print(f'For index:{index} range is:[{start_time}, {end_time}] \n')
        time_dimension.CumulVar(index).SetRange(start_time,end_time)

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        # AddVariableMinimizedByFinalizer creates a new variable in the solver model and specifies
        # that this variable should be minimized when the solver solves the model.
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





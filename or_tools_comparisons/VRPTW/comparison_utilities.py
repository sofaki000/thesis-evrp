from or_tools_comparisons.common_utilities import get_tour_length_from_distance_matrix
from or_tools_comparisons.VRPTW.VRPTW_ORTOOLS_model import create_data_model_vrptw, \
    solve_vrp_with_time_windows_with_or_tools
from or_tools_comparisons.CVRP.cvrp_or_tools import create_data_model, train_or_model_for_cvrp, print_solution
from plot_utilities import create_distance_matrix


def get_percentage_of_shorter_tours_VRPTW(test_loader, model):
    pointer_yper_or_tools = 0

    for batch in test_loader:
        static_elements, dynamic_elements, distance_matrix = batch

        time_windows = static_elements[:,2:]

        vrptw_tour_indices, _ ,_= model(static_elements ,dynamic_elements, distance_matrix)
        vrptw_tour_indices = vrptw_tour_indices[0]

        ##### OR TOOLS
        velocity = 10
        # h xronikh apostash ths kathe polis metaxy tous einai h kanonikh apostash dia thn taxhta oxhmatos
        time_matrix = distance_matrix / velocity
        data_for_or_model = create_data_model_vrptw(time_matrix=time_matrix[0,:,:],
                                                    time_windows= time_windows[0,:,:].transpose(1,0),
                                                    num_vehicles=1)

        counter_for_unsolved_problems = 0

        try:
            or_tour = solve_vrp_with_time_windows_with_or_tools(data_for_or_model)[0]
        except:
             print("Cant solve this problem statement")
             counter_for_unsolved_problems +=1
             continue

        print(f'Could not solve {counter_for_unsolved_problems} use cases.')



        # TODO: add to print solution
        # print("my model:\n")
        # model_output = print_solution(vrptw_tour_indices)
        # print("OR tools:\n")
        # or_output = print_solution(or_tour)
        distance_matrix = distance_matrix.squeeze()
        _, tour1_length = get_tour_length_from_distance_matrix(vrptw_tour_indices,distance_matrix)
        _, tour2_length = get_tour_length_from_distance_matrix(or_tour,distance_matrix)

        if tour2_length < tour1_length:
            # nikaei to or tools, exei shorter tour length
            pointer_yper_or_tools+=1


    # gyrname posa tours proteine to or tools shorter kai ena tour pou edwse to kathe mondelo endeiktika
    return pointer_yper_or_tools, vrptw_tour_indices, or_tour,static_elements,distance_matrix, time_windows
from or_tools_comparisons.common_utilities import get_tour_length_from_distance_matrix
from or_tools_comparisons.CVRP.cvrp_or_tools import create_data_model, train_or_model_for_cvrp
from ploting.plot_utilities import create_distance_matrix


def get_percentage_of_shorter_tours_CVRP(test_loader, model, max_load):
    pointer_yper_or_tools = 0

    for batch in test_loader:
        static_elements, dynamic_elements, _ = batch

        cvrp_tour_indices, _ = model(static_elements ,dynamic_elements)
        cvrp_tour_indices = cvrp_tour_indices[0]

        ##### OR TOOLS
        demands = dynamic_elements.squeeze(0)[1]
        distance_matrix = create_distance_matrix(static_elements.squeeze(0).transpose(1,0))
        data_for_or_model = create_data_model(distance_matrix=distance_matrix,
                                              demands=[demands],
                                              vehicle_capacities=[max_load])

        or_tour = train_or_model_for_cvrp(data_for_or_model)[0]

        # print("my model:\n")
        # model_output = print_solution(cvrp_tour_indices)
        # print("OR tools:\n")
        # or_output = print_solution(or_tour)

        _, tour1_length = get_tour_length_from_distance_matrix(cvrp_tour_indices,distance_matrix)
        _, tour2_length = get_tour_length_from_distance_matrix(or_tour,distance_matrix)

        if tour2_length < tour1_length:
            # nikaei to or tools, exei shorter tour length
            pointer_yper_or_tools+=1


    # gyrname posa tours proteine to or tools shorter kai ena tour pou edwse to kathe mondelo endeiktika
    return pointer_yper_or_tools, cvrp_tour_indices, or_tour,static_elements,distance_matrix
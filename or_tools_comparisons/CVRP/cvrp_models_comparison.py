
from torch.utils.data import DataLoader
from datasets.CVRP_dataset import CapacitatedVehicleRoutingDataset
from models.CVRP_SOLVER import get_trained_model_for_cvrp
from or_tools_comparisons.comparison_utilities import get_percentage_of_shorter_tours_CVRP
from ploting.plot_utilities import show_tour, get_filename_time



def compare_model_with_or_tools(model, model_type, test_loader, file_name_for_storing_results):
    number_of_tours_or_suggested_shorter_solutions, cvrp_tour_indices, or_tour, static_elements, distance_matrix \
        = get_percentage_of_shorter_tours_CVRP(test_loader, model, max_load)

    f = open(file_name_for_storing_results, "a")

    experiment_date = get_filename_time()

    percentage_of_or_shorter_tours = (number_of_tours_or_suggested_shorter_solutions * test_size) / 100
    content1 = f'Sta {test_size} exoume {number_of_tours_or_suggested_shorter_solutions} shorter paths proteinomena apo or tools\n'
    content2 = f'Dld to {percentage_of_or_shorter_tours}% twn tours proteinontai shorter apo or tools\n'
    f.write(f'\n------------------------{experiment_date}-------------------------\n')
    f.write(content1)
    print(content1)
    f.write(content2)
    print(content2)

    # blepoyme thn teleutaia diadromh

    show_tour(static_elements.squeeze(0).transpose(1, 0), distance_matrix,
              model_tour=cvrp_tour_indices, or_tour=or_tour,
              filename=f'{model_type}_vrp_nodes={num_nodes}_epochs={epochs}', model_tour_title=model_tour_title)

if __name__ == '__main__':
    testing = False
    if testing:
        test_size = 1
        epochs = 1
        num_nodes = 5
        train_size = 10
        validation_size = 10
    else:
        test_size = 150
        epochs = 15
        num_nodes = 5 #100
        train_size = 1000 #2000
        validation_size = 500 #1000

    batch_size = 25
    max_load = 20
    max_demand = 9

    train_ds = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes,max_load=max_load,max_demand=max_demand)
    validation_ds = CapacitatedVehicleRoutingDataset(num_samples=validation_size, input_size=num_nodes,max_load=max_load,max_demand=max_demand)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    # COMPARISON
    test_ds = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes, max_load=max_load,max_demand=max_demand)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)


    #### COMPARING WITH POINTER NETWORK
    file_name_for_storing_results_pntr = 'pntr_network_cvrp_comparison_with_or_tools.txt'
    model_tour_title = 'Tour from pointer network model'
    model_type_pntr= 'pntr'
    pointer_network_model = get_trained_model_for_cvrp(use_multihead_attention=False,
                                                       use_pointer_network=True,
                                                       epochs=epochs, train_loader=train_loader,
                                                        validation_loader=validation_loader)

    compare_model_with_or_tools(pointer_network_model, model_type_pntr, test_loader, file_name_for_storing_results_pntr)

    #### COMPARING WITH MULTIHEAD ATTENTION MODEL
    file_name_for_storing_results_multihead = 'multihead_network_cvrp_comparison_with_or_tools.txt'
    model_tour_title= 'Tour from multihead attention model'
    model_type_multihead = 'multihead'
    multihead_attention_model = get_trained_model_for_cvrp(use_multihead_attention=True,
                                                           use_pointer_network=False,
                                                           epochs=epochs,
                                                           train_loader=train_loader,
                                                           validation_loader=validation_loader)
    compare_model_with_or_tools(multihead_attention_model, model_type_multihead, test_loader, file_name_for_storing_results_multihead)

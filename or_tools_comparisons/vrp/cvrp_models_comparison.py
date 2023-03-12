
from torch.utils.data import DataLoader
from datasets.capacitated_vrp_dataset import CapacitatedVehicleRoutingDataset
from cvrp_main import train_cvrp_model_pntr
from models.CVRP_SOLVER import get_trained_model_for_cvrp
from or_tools_comparisons.comparison_utilities import get_percentage_of_shorter_tours
from plot_utilities import show_tour, get_filename_time
testing = True

if __name__ == '__main__':
    if testing:
        test_size = 1
        epochs = 1
        num_nodes = 5
        train_size = 10
        validation_size = 10
    else:
        test_size = 150
        epochs = 15
        num_nodes = 100
        train_size = 2000
        validation_size = 1000

    batch_size = 25
    max_load = 20
    max_demand = 9

    train_ds = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes,max_load=max_load,max_demand=max_demand)
    validation_ds = CapacitatedVehicleRoutingDataset(num_samples=validation_size, input_size=num_nodes,max_load=max_load,max_demand=max_demand)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    # use_multihead_attention = True #False
    # use_pointer_network = False #True
    use_multihead_attention = False
    use_pointer_network = True

    cvrp_model = get_trained_model_for_cvrp(use_multihead_attention, use_pointer_network, epochs, train_loader, validation_loader)
    #cvrp_model = train_cvrp_model(epochs, train_loader ,validation_loader)

    # COMPARISON
    test_ds = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes,
                                               max_load=max_load,max_demand=max_demand)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)


    number_of_tours_or_suggested_shorter_solutions, cvrp_tour_indices, or_tour, static_elements, distance_matrix\
        = get_percentage_of_shorter_tours(test_loader, cvrp_model, max_load)

    f = open("cvrp_comparison.txt", "a")

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
    if use_pointer_network:
        model_tour_title = 'Tour from pointer network model'
    elif use_multihead_attention:
        model_tour_title= 'Tour from multihead attention model'

    show_tour(static_elements.squeeze(0).transpose(1,0), distance_matrix,
              model_tour=cvrp_tour_indices, or_tour=or_tour,
              filename=f'vrp_nodes={num_nodes}_epochs={epochs}', model_tour_title=model_tour_title)
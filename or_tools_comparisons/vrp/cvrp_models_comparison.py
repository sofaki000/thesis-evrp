
from torch.utils.data import DataLoader
from datasets.capacitated_vrp_dataset import CapacitatedVehicleRoutingDataset
from cvrp_main import train_cvrp_model
from cvrp_or_tools import create_data_model, train_or_model_for_cvrp
from plot_utilities import create_distance_matrix, show_tour

from or_tools_comparisons.common_utilities import print_solution

if __name__ == '__main__':
    epochs = 1
    num_nodes = 4
    train_size = 100
    test_size = 10
    batch_size = 25
    max_load = 20
    max_demand = 9

    train_dataset = CapacitatedVehicleRoutingDataset(num_samples=train_size, input_size=num_nodes,max_load=max_load,max_demand=max_demand)
    test_dataset = CapacitatedVehicleRoutingDataset(num_samples=test_size, input_size=num_nodes,max_load=max_load,max_demand=max_demand)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


    cvrp_model = train_cvrp_model(epochs, train_loader ,validation_loader)

    ########## We compare the two models for a sample
    static_elements, dynamic_elements, _ = train_dataset.__getitem__(0)

    cvrp_tour_indices, _ = cvrp_model(static_elements.unsqueeze(0),dynamic_elements.unsqueeze(0))

    ##### OR TOOLS
    demands = dynamic_elements[1]
    load = 1
    distance_matrix = create_distance_matrix(static_elements.transpose(1,0))
    data_for_or_model = create_data_model(distance_matrix=distance_matrix,
                                          demands=[demands],
                                          vehicle_capacities=[max_load])

    or_tour = train_or_model_for_cvrp(data_for_or_model)

    print("my model:\n")
    model_output = print_solution(cvrp_tour_indices[0])

    print("OR tools:\n")
    or_output = print_solution(or_tour[0])


    show_tour(static_elements.transpose(1,0), distance_matrix,
              model_tour=cvrp_tour_indices[0], or_tour=or_tour[0], filename='vrp_first')
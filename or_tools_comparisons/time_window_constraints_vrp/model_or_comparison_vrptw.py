import torch
from torch.utils.data import DataLoader
from datasets.VRPTW_dataset import VRPTW_data_simple
from or_tools_comparisons.time_window_constraints_vrp.VRTW_SOLVER import VRPTW_SOLVER_MODEL
from or_tools_comparisons.time_window_constraints_vrp.comparison_utilities import get_percentage_of_shorter_tours_VRPTW
from or_tools_comparisons.time_window_constraints_vrp.main import train_vrptw_model
from or_tools_comparisons.time_window_constraints_vrp.time_windows_utilities import create_gaant_chart_from_times
from plot_utilities import show_tour, get_filename_time


def compare_model_with_or_tools_VRPTW(model, test_loader, file_name_for_storing_results):
    number_of_tours_or_suggested_shorter_solutions, cvrp_tour_indices, or_tour, static_elements, distance_matrix, time_windows\
        = get_percentage_of_shorter_tours_VRPTW(test_loader, model)

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
    locations = static_elements[:,0:2].squeeze(0).transpose(1, 0)

    start_times = time_windows[:,0].squeeze(0).numpy()
    end_times = time_windows[:,1].squeeze(0).numpy()
    create_gaant_chart_from_times(start_times, end_times)

    show_tour(locations,distance_matrix,
              model_tour=cvrp_tour_indices, or_tour=or_tour,
              filename=f'VRPTW_nodes={num_nodes}_epochs={epochs}',
              model_tour_title=model_tour_title)



if __name__ == '__main__':
    testing = False #True
    if testing:
        print("TRAINING FOR VERY FEW EPOCHS")
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

    train_dataset = VRPTW_data_simple(train_size, num_nodes)
    validation_dataset = VRPTW_data_simple(validation_size, num_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # COMPARISON
    test_dataset = VRPTW_data_simple(train_size, num_nodes)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)


    #### COMPARING WITH POINTER NETWORK
    file_name_for_storing_results_pntr = 'VRPTW_model_comparison_with_or_tools.txt'
    model_tour_title = 'Tour from model'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = VRPTW_SOLVER_MODEL()

    use_trained_model = True #False #
    PATH = "./model_vrptw.pt"

    if use_trained_model:
       print("Loading pretrained model...")
       model.load_state_dict(torch.load(PATH))
    else:
        print("Training model...")
        model = train_vrptw_model(model, epochs, train_loader, validation_loader)
        torch.save(model.state_dict(), PATH)


    compare_model_with_or_tools_VRPTW(model,  test_loader, file_name_for_storing_results_pntr)



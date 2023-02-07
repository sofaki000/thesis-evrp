import numpy as np

from plot_utilities import show_tour, get_filename_time
from tsp_main import train_tsp_model
from tsp_or_tools import run_or_model, create_data_model
from dataset import TSPDataset
from torch.autograd import Variable

f = open("comparison_tsp.txt", "a")
f.write(f'Experiment at:{get_filename_time()}\n')

def get_tour_length_from_distance_matrix(tour, distance_matrix):
    length = 0
    prev_index = tour[0]
    for i in range(len(tour)):
        index = tour[i]
        length += distance_matrix[index][prev_index]
        prev_index = index

    content = f'Distance travelled:{length}'
    print("----------------------")
    print(content)
    return content, length

def print_solution(tour):
    plan_output = 'From my model:\n'
    for i in range(len(tour)):
        if i == len(tour-1):
            plan_output += ' {}'.format(tour[i])
        else:
            plan_output += ' {} ->'.format(tour[i])
    print(plan_output )
    return plan_output

if __name__ == '__main__':
    epochs = 30
    num_nodes = 6
    train_size = 10
    test_size = 5
    train_dataset = TSPDataset(train_size, num_nodes)
    test_dataset = TSPDataset(test_size, num_nodes)
    my_model, outputs = train_tsp_model(train_dataset, test_dataset,epochs)

    percentage_a_tour_is_better = 0
    number_of_tours_to_make_comparison = 1

    for i in range(number_of_tours_to_make_comparison):
        problem = train_dataset.__getitem__(i)
        nodes = Variable(problem['Points']).unsqueeze(0)

        target_batch = Variable(problem['Solution'])
        distance_matrix = problem['Distance_matrix']
        tour_from_my_model = my_model(nodes, target_batch)

        tour1 = np.append(tour_from_my_model[1].detach().squeeze().numpy(),0)
        model_tours = print_solution(tour1)
        f.write("Model:")
        content_from_my_model, tour_length1 = get_tour_length_from_distance_matrix(tour1, distance_matrix)
        f.write(model_tours)
        f.write(content_from_my_model)
        data = create_data_model(distance_matrix)
        or_routes,or_tools_results = run_or_model(data)
        f.write("OR TOOLS\n")
        tour2 = or_routes[0]
        content_from_or, tour_length2 = get_tour_length_from_distance_matrix(tour2, distance_matrix)
        distance_travelled= or_tools_results

        f.write(content_from_or)
        f.write(distance_travelled)

        show_tour(nodes.squeeze(0), distance_matrix, tour1, tour2, filename=f'epochs={epochs}_nodes={num_nodes}')

        percentage_a_tour_is_better += (tour_length1 - tour_length2) / tour_length1
        f.write('--------------\n\n')

    f.write(f"Finally: OR tools has {percentage_a_tour_is_better*100/number_of_tours_to_make_comparison} shorter solutions")

    f.close()




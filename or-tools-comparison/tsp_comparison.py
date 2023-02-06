from tsp_main import train_tsp_model
from tsp_or_tools import run_or_model, create_data_model
from dataset import TSPDataset
from torch.autograd import Variable
import torch

f = open("comparison_tsp.txt", "a")

def get_tour_length_from_distance_matrix(tour, distance_matrix):

    length = 0
    prev_index = tour[0]
    for i in range(len(tour)):
        index = tour[i]
        length += distance_matrix[index][prev_index]
        prev_index = index

    content = f'Distance travelled:{length}'
    print(content)

    return content, length

def print_solution(tour_from_my_model):
    tour = tour_from_my_model[1].detach().squeeze().numpy()
    plan_output = 'From my model: Route for vehicle 0:\n'
    for i in range(len(tour)):
        if i == len(tour-1):
            plan_output += ' {}'.format(tour[i])
        else:
            plan_output += ' {} ->'.format(tour[i])
    print(plan_output )
    return plan_output

if __name__ == '__main__':
    num_nodes = 13
    train_size = 10
    test_size = 5
    train_dataset = TSPDataset(train_size, num_nodes)
    test_dataset = TSPDataset(test_size, num_nodes)
    my_model, outputs = train_tsp_model(train_dataset, test_dataset)


    percentage_a_tour_is_better = 0

    for i in range(10):
        problem = train_dataset.__getitem__(1)
        train_batch = Variable(problem['Points'])
        target_batch = Variable(problem['Solution'])
        distance_matrix = problem['Distance_matrix']
        tour_from_my_model = my_model(train_batch.unsqueeze(0),target_batch)

        model_tours = print_solution(tour_from_my_model)
        f.write("Model:")
        content_from_my_model, tour1=get_tour_length_from_distance_matrix(outputs[0], distance_matrix)
        f.write(model_tours)
        f.write(content_from_my_model)
        data = create_data_model(distance_matrix)
        or_routes,or_tools_results = run_or_model(data)
        f.write("OR TOOLS\n")

        content_from_or, tour2 = get_tour_length_from_distance_matrix(or_routes[0], distance_matrix)
        distance_travelled= or_tools_results

        f.write(content_from_or)
        f.write(distance_travelled)

        percentage_a_tour_is_better += (tour1-tour2)/tour1
        f.write('--------------\n\n')

    f.write(f"Finally: OR tools has {percentage_a_tour_is_better*100/10} shorter solutions")

    f.close()




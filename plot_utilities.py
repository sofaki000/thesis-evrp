import torch
import math
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix

def create_distance_matrix(points):
    return distance_matrix(points, points)

from config import losses_dir,rewards_dir


def get_filename_time():
    now = datetime.datetime.now()
    return f'm={now.month}_d={now.day}_h={now.hour}_m={now.minute}'


def show_tour_for_model(ax, distance_matrix, nodes, tour):
    N = len(nodes)
    distance =0.
    start_node = 0

    for i in range(N):
        start_pos = nodes[start_node]
        if i != N-1:
            next_node = tour[i + 1]
        else:
            next_node = tour[0]

        end_pos = nodes[next_node]
        ax.annotate("",xy=end_pos, xycoords='data',
                           xytext=start_pos , textcoords='data',
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        #print(f'{start_node}->{next_node}, dist:{distance_matrix[start_node][next_node]}')
        distance += distance_matrix[start_node][next_node] #math.dist(end_pos, start_pos)
        start_node = next_node
        if torch.is_tensor(end_pos[0]):
            if torch.is_tensor(next_node):
                ax.text(end_pos[0].item(), end_pos[1].item(), next_node.item(), size=10, color='b')
            else:
                ax.text(end_pos[0].item(), end_pos[1].item(), next_node, size=10, color='b')
        else:
            ax.text(end_pos[0], end_pos[1], next_node, size=10, color='b')

    textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,  # Textbox
               verticalalignment='top', bbox=props)

def get_2city_distance(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError

def show_tour(nodes, distance_matrix, model_tour, or_tour, filename):
    '''
    nodes: tensor [num_nodes, 2]
    model_tour, or_tour: [num_nodes+1]
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, sharex=True, sharey=True)  # Prepare 2 plots
    ax[0].set_title('OR tour')
    ax[0].scatter(nodes[:, 0], nodes[:, 1])  # plot A
    show_tour_for_model(ax[0],distance_matrix, nodes, or_tour)


    ax[1].set_title('Tour from model')
    ax[1].scatter(nodes[:, 0], nodes[:, 1])  # plot B
    show_tour_for_model(ax[1],distance_matrix, nodes, model_tour)

    plt.tight_layout()
    plt.savefig(f"tour_1{filename}.png")



def plot_losses_and_rewards(losses_per_epochs, rewards_per_epochs):
    time = get_filename_time()
    # epochs finished
    plt.plot(losses_per_epochs)
    plt.savefig(f"{losses_dir}losses{time}.png")
    plt.clf()
    plt.plot(rewards_per_epochs)
    plt.savefig(f"{rewards_dir}rewards{time}.png")
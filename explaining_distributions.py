import matplotlib.pyplot as plt
import numpy as np
import torch

from ploting.plot_utilities import create_distance_matrix

probs = torch.randint(10, (1,4)) #torch.ones(2,5)



def get_distance_of_each_city_from_current_city(static, current_position):
    ''' static: shape [2, num_nodes] '''
    distance_matrix = create_distance_matrix(static.transpose(1,0))

    return distance_matrix[:, current_position]

# IDEA: Se kathe polh na exeis kapoia xarakthristika
# px posh apostash exei apo ekei pou eimaste, ti load kai demand exoume

def explain_decision(static, dynamic, distribution, current_tour,chosen_city):

    tour = current_tour.detach().numpy()

    current_position = tour[-1]

    distance_of_each_city_from_current_city = get_distance_of_each_city_from_current_city(static, current_position)

    plot_categorical_distribution_with_descriptions_at_categories(distribution,dynamic,  chosen_city,distance_of_each_city_from_current_city)


def plot_categorical_distribution_with_descriptions_at_categories(categorical_distribution,dynamic, chosen_city, distance_of_each_city_from_current_city):
    probs = categorical_distribution.probs

    probs= probs.detach().numpy()[0]
    categories = range(len(probs))
    fig, ax = plt.subplots()
    ax.bar(categories, probs)
    # ax.set_xlabel('Potential cities to visit')
    # ax.set_ylabel('Probability to visit each city')
    ax.set_xlabel('Πόλεις που μπορεί να επισκεφτεί')
    ax.set_ylabel('Πιθανότητα να επισκεφτεί κάθε πόλη')
    ax.set_xticks(categories)

    load = dynamic[:,0][0].item()
    ax.set_title(f"Φορτίο:{load:.2f}", fontsize=8, y=1.08)
    for i, prob in enumerate(probs):
        current_bar_dynamics = dynamic[:,i]

        demand = current_bar_dynamics[1].item()
        text = f"Απόσταση: {distance_of_each_city_from_current_city[i]:.2f}\n" #{prob:.2f}"
        if demand< 0:
            demand = 0
        text += f'Ζήτηση:{demand:.2f}\n'

        if chosen_city.item() == i:
            text += "\nΕπιλογή"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.annotate(text, (i, prob), ha='center', va='bottom', fontsize=7, bbox=props)


    fig.tight_layout()
    plt.savefig("distribution.png")

def plot_distribution(distribution, title='Distribution Plot'):
    probabilities = distribution.probs.detach().numpy()[0]
    categories = np.arange(len(probabilities))
    plt.bar(categories, probabilities)
    plt.xlabel('Categories')
    plt.ylabel('Probabilities')
    plt.title(title)


    plt.show()





# distrb = torch.distributions.Categorical(probs)
# plot_distribution(distrb)


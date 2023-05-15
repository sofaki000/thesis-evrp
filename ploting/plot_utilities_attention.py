import os

import matplotlib.pyplot as plt
import seaborn as sns

from ploting.plot_utilities import get_filename_time

attention_weights_dir = "attention_weights_plots"
os.makedirs(attention_weights_dir, exist_ok=True)


def get_attention_weights_dir():
    return attention_weights_dir
def get_next_experiment_number(directory):
    prefix = "exp"
    existing_numbers = []

    # Iterate over the directories in the given directory
    for entry in os.scandir(directory):
        if entry.is_dir() and entry.name.startswith(prefix):
            try:
                number = int(entry.name[len(prefix):])
                existing_numbers.append(number)
            except ValueError:
                pass

    # Find the next available number
    next_number = 1
    while next_number in existing_numbers:
        next_number += 1

    return next_number

def plot_attention_weights_heatmap_for_each_timestep(attention_weights,
                                                     experiment_name,
                                                     experiment_folder_name,
                                                     epoch):
    """
    Plots attention weights as a heatmap.

    Args:
        attention_weights (list): List of attention weights for each step,
         shape (n_steps, n_locations)

    Returns:
        None
    """
    # Create a heatmap using seaborn
    sns.heatmap(attention_weights, cmap="YlGnBu", annot=True, cbar=True)
    plt.xlabel("Locations")
    plt.ylabel("Steps")
    plt.title(f"Attention Weights Heatmap, Epoch {epoch}")

    os.makedirs(f"{attention_weights_dir}\\{experiment_folder_name}", exist_ok=True)

    plt.savefig(f"{attention_weights_dir}\\{experiment_folder_name}\\attention_weights_EPOCH{epoch}___{experiment_name}_{get_filename_time()}.png")

    plt.clf()


def plot_heatmap_with_attention_weights_over_time(attention_weights):
    '''
    attention_weights: 2D array of attention weights for each time step and input element
    '''
    import matplotlib.pyplot as plt

    # Replace with your extracted attention weights

    # Create a heatmap to visualize the attention weights over time
    plt.figure(figsize=(10, 6))
    plt.imshow(attention_weights, cmap='hot', interpolation='nearest', aspect='auto')
    plt.xlabel('Input Element')
    plt.ylabel('Time Step')
    plt.title('Attention Weights over Time')
    plt.colorbar()
    plt.savefig("heatmap_with_attention_weights_over_time.png")


def plot_attention_weights(attention_weights, seq_len):
    import matplotlib.pyplot as plt

    # Assume you have attention_weights as a list of attention weights for each time step
    #attention_weights = [0.1, 0.3, 0.4, 0.2, 0.0]

    # Choose the index for which you want to plot the attention weights
    chosen_index = 2

    # Prepare the input sequence
    input_sequence = [i for i in range(0, seq_len)] #[1, 2, 3, 4, 5]

    # Plot the attention weights for the chosen index
    plt.plot(input_sequence, attention_weights)
    plt.scatter(input_sequence[chosen_index],
                attention_weights[chosen_index],
                c='r', marker='o', label='Chosen Index')
    plt.xlabel('Input Sequence')
    plt.ylabel('Attention Weights')
    plt.title('Attention Weights for Chosen Index')
    plt.legend()
    plt.savefig("attention_weights.png")
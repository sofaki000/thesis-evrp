import os

import matplotlib.pyplot as plt

import datetime
from ploting.plot_utilities import get_filename_time


def plot_average_tour_length(tour_lengths,experiment_details):
    """
    Plots the average tour length given a list of tour lengths.

    Parameters:
        tour_lengths (list): A list of tour lengths.
    Returns:
        None (displays the plot)

    Example:
        tour_lengths = [10, 12, 8, 15, 11, 13, 9]
    """
    if not tour_lengths:
        print("Error: The list of tour lengths is empty.")
        return

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(tour_lengths)
    plt.xlabel('Tour')
    plt.ylabel('Tour Length')
    plt.title('Average tour length per epoch')
    plt.legend(['Average Tour Length'])
    plt.grid(True)
    plt.tight_layout()

    filename = f"avgTourLen_{experiment_details}_date{get_filename_time()}.png"

    now = datetime.datetime.now()
    directory = f'metrics\\avg_tour_length\\day_{now.day}\\hour_{now.hour}'

    os.makedirs(directory, exist_ok=True)

    plt.savefig(f'{directory}\\{filename}')

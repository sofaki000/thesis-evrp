import random

import matplotlib.pyplot as plt
import numpy as np
import torch

def get_customer_names_for_gantt(start_times):
    num_customers = len(start_times)
    customer_names = []

    for i in range(num_customers):
        customer_names.append(f'Customer {i}')

    return customer_names

def create_gaant_chart_from_times(start_times, end_times):
    # Create a horizontal bar chart

    task_names = get_customer_names_for_gantt(start_times)
    fig, ax = plt.subplots(figsize=(8, 3))

    # Set the chart title and axes labels
    ax.set_title('Gantt Chart')
    ax.set_xlabel('Hours')
    ax.set_yticks(np.arange(len(task_names)))
    ax.set_yticklabels(task_names)

    # Set the x-axis limits and ticks

    min_tick = np.min(start_times)
    max_tick = np.max(end_times)+1
    ax.set_xlim(min_tick, max_tick)
    ax.set_xticks(np.arange(min_tick, max_tick))

    # Plot the tasks as horizontal bars
    for i in range(len(task_names)):
        ax.barh(i, end_times[i] - start_times[i], left=start_times[i])

    plt.savefig("gaant.png")


def get_start_and_end_times():
    start_times = [0, 2, 3, 5, 6]
    end_times = [12, 16, 18, 24, 30]
    return start_times, end_times


def get_batch_for_time_intervals(train_size):
    time_windows_batch = []
    start_times = [0, 2, 3, 5, 6]
    end_times = [12, 16, 18, 24, 30]

    for i in range(train_size):
        # get a random number
        r = random.randint(0, 10)

        # add it to interval
        start_times = np.array(start_times) + r
        end_times = np.array(end_times) + r

        # add new interval to result
        earliest_window = torch.tensor(start_times)

        latest_window = torch.tensor(end_times)

        time_windows = torch.cat((earliest_window.unsqueeze(0), latest_window.unsqueeze(0)))

        time_windows_batch.append(time_windows.unsqueeze(0))

    return torch.cat(time_windows_batch, 0)

# # Define task names and their start and end times
# task_names = ['Customer 1', 'Customer 2', 'Customer 3', 'Customer 4', 'Customer 5']
# start_times, end_times = get_start_and_end_times()
# create_gaant_chart_from_times(start_times, end_times)
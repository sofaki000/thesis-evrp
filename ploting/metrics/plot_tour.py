import numpy as np
from tabulate import tabulate

# Given tensors
target = np.array([[0, 1, 2],
                   [0, 1, 2]])

model_result = np.array([[1, 0, 2],
                         [0, 2, 1]])

# Function to print tensors nicely using tabulate
def print_tensor(tensor):
    batch_size = tensor.size(0)
    seq_len = tensor.size(1)

    for batch in range(batch_size):
        print(f'Batch: {batch}: ', end = '')
        for seq in range(seq_len):
            if seq== seq_len-1:
                print(f'{tensor[batch][seq]}', end='')
            else:
                print(f'{tensor[batch][seq]} -> ', end = '')

        print("\n")




def format_tour(  model_tour):
    """
    Formats the target tour and model result tour side by side.
    Parameters:

        model_tour (list): List of indices representing the model result tour.
    """
    print("Model Result Tensor:")
    print_tensor(model_tour)


def format_tours(target_tour, model_tour):
    """
    Formats the target tour and model result tour side by side.
    Parameters:
        target_tour (list): List of indices representing the target tour.
        model_tour (list): List of indices representing the model result tour.
    """
    print("Target Tensor:")
    print_tensor(target_tour)
    # Print model result tensor
    print("Model Result Tensor:")
    print_tensor(model_tour)

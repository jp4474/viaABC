import numpy as np
import os
from matplotlib import pyplot as plt

def plot_simulation(data: dict, channels: int, channel_names: list, rows: int = 1):
    """
    Plots simulation data in a grid of subplots.

    Parameters:
        data (dict): Dictionary containing simulation data.
        channels (int): Number of channels to plot.
        channel_names (list): List of channel names.
        rows (int): Number of rows in the grid.
    """
    simulation = data['simulations']
    idx = np.random.randint(0, simulation.shape[0], size=rows)

    # Create figure with specified size
    fig, axs = plt.subplots(rows, channels, figsize=(channels * 5, rows * 5))

    # Ensure axs is always a 2D array, even when rows == 1
    if rows == 1:
        axs = np.array([axs])

    # Plot each channel
    for i, sim_idx in enumerate(idx):
        for j in range(channels):
            ax = axs[i, j]  # Access the subplot correctly
            ax.plot(simulation[sim_idx, :, j])
            ax.set_title(channel_names[j])
            ax.grid(True)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Amplitude')

    # Adjust spacing between subplots
    plt.tight_layout()

    plt.show()
    plt.close()

def plot_parameters(data: dict, num_parameters: int, channel_names: list, rows: int = 1):
    """
    Plots parameter data in a grid of subplots.

    Parameters:
        num_parameters (int): Number of parameters to plot.
        channel_names (list): List of parameter names.
        rows (int): Number of rows in the grid.
    """
    parameters = data['params']

    # Create figure with specified size
    fig, axs = plt.subplots(1, num_parameters, figsize=(10,  5))

    # overlay plots with normal distribution with mean and std
    # loc = [0.5, 14, 0.1, -5, 3.5, -4]
    # scale = [0.25, 2, 0.15, 1.2, 0.8, 1.2]

    
    for j in range(num_parameters):
        ax = axs[j]
        # Plotting only the "Accepted" data histogram.
        ax.hist(parameters[:, j], bins=30, density=True, label='Accepted')
        ax.set_title(channel_names[j])
        ax.grid(True)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    plt.show()
    plt.close()
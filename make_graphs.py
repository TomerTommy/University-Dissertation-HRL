import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth_values(values, weight=0.3):
    """
    Apply exponential moving average smoothing.
    
    Parameters:
        values (np.array): The array of values to smooth.
        weight (float): The smoothing weight (0 < weight < 1).
        
    Returns:
        np.array: Smoothed values.
    """
    last = values[0]  # First value in the smoothed array
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_tensorboard_log(log_paths, labels, colors, metric_name, save_path):
    """
    Plots a graph from TensorBoard logs.

    Parameters:
        log_path (str): The path to the TensorBoard log files.
        metric_name (str): The name of the metric to plot (e.g., 'loss', 'accuracy').
        save_path (str): The path to save the resulting plot.
    """
    
    plt.figure(figsize=(10, 5))
    for path_idx in range(len(log_paths)):
        # Initialize an event accumulator
        event_acc = EventAccumulator(log_paths[path_idx], size_guidance={'scalars': 0})
        event_acc.Reload()

        if metric_name not in event_acc.Tags()['scalars']:
            assert f'The metric {metric_name} is not found in the logs.'

        # Extract and plot the scalar data
        scalar_data = event_acc.Scalars(metric_name)

        steps = np.array([s.step for s in scalar_data])
        values = np.array([s.value for s in scalar_data])
        
        smoothed_values = smooth_values(values)

        plt.plot(steps, smoothed_values, color=colors[path_idx], label=labels[path_idx])
    plt.xlabel('Timesteps')
    plt.ylabel('Negative Losses')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

# usage
log_directory = ['experiments/runs/diayn_ppo_bpwhc_base_40m__1712247064', 'experiments/runs/diayn_ppo_bpw_base_40m__1712246936']
labels = ['Hardcore', 'Basic']
colors = ['blue', 'orange']
metric = 'losses/discriminator_loss_per_global'
output_file = 'experiments/matplotlib_graphs/diayn_base_40m_discriminator_losses_pyplot.png'

plot_tensorboard_log(log_directory, labels, colors, metric, output_file)

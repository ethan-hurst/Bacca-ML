# utils.py
import os
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

def create_weights_directory(weights_dir='weights'):
    """Creates the weights directory if it doesn't exist."""
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir

def initialize_tensorboard(log_dir='logs'):
    """Initializes a TensorBoard summary writer."""
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.summary.create_file_writer(log_dir)

def save_aggregate_profit_plot(aggregate_profits, filename='aggregate_profit.png'):
    """Saves a plot of the aggregate profits over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(aggregate_profits) + 1), aggregate_profits, marker='o')
    plt.title('Aggregate Profit Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Profit')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_agent_weights(agents, weights_dir='weights'):
    """Saves the weights of all agents."""
    for idx, agent in enumerate(agents):
        weights_file = os.path.join(weights_dir, f'agent_{idx}.weights.h5')
        agent.model.save_weights(weights_file)

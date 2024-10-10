# utils.py
import os
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

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
    """Saves the weights of all agents concurrently."""
    def save_single_agent(agent, idx):
        weights_file = os.path.join(weights_dir, f'agent_{idx}.weights.h5')
        agent.save_weights(weights_file)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_single_agent, agent, idx) for idx, agent in enumerate(agents)]
        for future in futures:
            future.result()

def log_metrics_to_tensorboard(summary_writer, generation, total_profit, max_profit, min_profit, average_profit, num_survivors, epsilon_values):
    """Logs metrics to TensorBoard."""
    with summary_writer.as_default():
        tf.summary.scalar('Total Profit', total_profit, step=generation)
        tf.summary.scalar('Max Profit', max_profit, step=generation)
        tf.summary.scalar('Min Profit', min_profit, step=generation)
        tf.summary.scalar('Average Profit', average_profit, step=generation)
        tf.summary.scalar('Number of Survivors', num_survivors, step=generation)
        for idx, epsilon in enumerate(epsilon_values):
            tf.summary.scalar(f'Agent_{idx}_Epsilon', epsilon, step=generation)
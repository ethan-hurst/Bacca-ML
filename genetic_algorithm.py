# genetic_algorithm.py
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

# Mutation function for genetic algorithm
def mutate_weights(weights, mutation_rate=0.01):
    mutated_weights = []
    for w in weights:
        # Apply Gaussian noise to weights for mutation
        noise = np.random.normal(0, mutation_rate, w.shape)
        mutated_weights.append(w + noise)
    return mutated_weights

# Crossover function for genetic algorithm
def crossover_weights(parent1_weights, parent2_weights):
    new_weights = []
    for w1, w2 in zip(parent1_weights, parent2_weights):
        # Uniform crossover: randomly choose weights from either parent
        mask = np.random.randint(0, 2, size=w1.shape).astype(bool)
        new_weight = np.where(mask, w1, w2)
        new_weights.append(new_weight)
    return new_weights

# Function to create new agents through mutation and crossover
def create_new_agents(alive_agents, num_new_agents, state_size, action_size, agent_class, mutation_rate=0.01, adaptive_exploration=False):
    new_agents = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(num_new_agents):
            futures.append(executor.submit(create_single_agent, alive_agents, state_size, action_size, agent_class, mutation_rate, adaptive_exploration))
        for future in futures:
            new_agents.append(future.result())
    return new_agents

# Helper function to create a single agent
def create_single_agent(alive_agents, state_size, action_size, agent_class, mutation_rate, adaptive_exploration):
    # Select two parents for crossover
    parent1, parent2 = random.sample(alive_agents, 2)
    parent1_weights = parent1[1].model.get_weights()
    parent2_weights = parent2[1].model.get_weights()

    # Perform crossover and mutation
    child_weights = crossover_weights(parent1_weights, parent2_weights)
    mutated_weights = mutate_weights(child_weights, mutation_rate)

    # Create new agent with mutated weights
    new_agent = agent_class(state_size, action_size)

    # Adaptive exploration adjustment
    if adaptive_exploration:
        # Agents with a high balance should have a lower epsilon to refine strategies
        # while still maintaining some exploration.
        if parent1[2] > 150 and parent2[2] > 150:
            new_agent.epsilon = min(parent1[1].epsilon, parent2[1].epsilon) * 0.8  # Lower exploration for high bankroll
        # Agents with a low balance should increase exploration in hopes of recovery.
        elif parent1[2] < 50 or parent2[2] < 50:
            new_agent.epsilon = max(parent1[1].epsilon, parent2[1].epsilon) * 1.2  # Higher exploration for low bankroll
        else:
            new_agent.epsilon = (parent1[1].epsilon + parent2[1].epsilon) / 2  # Average epsilon otherwise
    new_agent.model.set_weights(mutated_weights)
    new_agent.target_model.set_weights(mutated_weights)
    return new_agent

# Elitism function to retain the best agents
def retain_best_agents(alive_agents, num_to_retain):
    # Sort agents by their final balance in descending order
    alive_agents.sort(key=lambda x: x[2], reverse=True)
    # Retain the top agents
    return alive_agents[:num_to_retain]
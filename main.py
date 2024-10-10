# main.py
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from bacca_env import BaccaEnv
from dqn_agent import DQNAgent
from genetic_algorithm import create_new_agents

# Create the weights directory if it doesn't exist
weights_dir = 'weights'
os.makedirs(weights_dir, exist_ok=True)

# Set parameters
history_length = 100
state_size = 2 + history_length
action_size = 3
population_size = 100
generations = 250
shoes_per_generation = 1

# Initialize agents
agents = []
survival_streaks = [0] * population_size  # Track survival streaks for each agent
for idx in tqdm(range(population_size), desc="Loading Agents"):
    agent = DQNAgent(state_size, action_size)
    weights_file = os.path.join(weights_dir, f'agent_{idx}.weights.h5')
    if os.path.exists(weights_file):
        try:
            agent.model.load_weights(weights_file)
            agent.target_model.load_weights(weights_file)
        except Exception as e:
            print(f"Error loading weights for agent {idx + 1}: {e}")
    agents.append(agent)

# Initialize the shared environment
env = BaccaEnv(num_agents=population_size, history_length=history_length)

# Initialize a list to store aggregate profits
aggregate_profits = []

# Initialize TensorBoard writer
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer = tf.summary.create_file_writer(log_dir)

try:
    for generation in range(generations):
        print(f"\n--- Starting Generation {generation + 1}/{generations} ---\n")

        # Ensure balances are not reset between shoes
        for shoe_number in range(shoes_per_generation):
            print(f"--- Starting Shoe {shoe_number + 1}/{shoes_per_generation} in Generation {generation + 1} ---")

            states = env.reset(reset_balances=False)
            done = False
            while not done:
                actions = []
                bet_sizes = []
                for idx, agent in enumerate(agents):
                    state = states[idx]
                    if env.balances[idx] >= min(env.allowed_bet_units):
                        action, bet_size = agent.act(state, env.balances[idx], env.sit_out_counts[idx])
                    else:
                        action = 0  # Default action
                        bet_size = 0
                    actions.append(action)
                    bet_sizes.append(bet_size)

                next_states, rewards, done, _ = env.step(actions, bet_sizes)

                for idx, agent in enumerate(agents):
                    state = states[idx]
                    action = actions[idx]
                    reward = rewards[idx]
                    next_state = next_states[idx]
                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) >= 256 and random.random() < 0.1:
                        agent.replay(64)

                states = next_states

            print(f"Completed Shoe {shoe_number + 1} in Generation {generation + 1}")

        alive_agents = []
        bankrupt_count = 0
        survivor_profits = []

        for idx, agent in enumerate(agents):
            final_balance = env.balances[idx]
            initial_balance = env.initial_balance
            profit = final_balance - initial_balance
            if final_balance >= min(env.allowed_bet_units):
                alive_agents.append((idx, agent, final_balance))
                survivor_profits.append(profit)
            else:
                bankrupt_count += 1
                survival_streaks[idx] = 0  # Reset streak for bankrupt agents

        print(f"Total bankrupt agents: {bankrupt_count}")

        if survivor_profits:
            total_profit = sum(survivor_profits)
            max_profit = max(survivor_profits)
            min_profit = min(survivor_profits)
            average_profit = total_profit / len(survivor_profits)
            print(f"Generation {generation + 1} Survivors' Profits:")
            # Print the top 5 performers
            top_performers = sorted(alive_agents, key=lambda x: ((x[2] - (env.initial_balance if survival_streaks[x[0]] == 0 else env.balances[x[0]])) / (env.initial_balance if survival_streaks[x[0]] == 0 else env.balances[x[0]])) * 100, reverse=True)[:5]
            print("Top 5 Performers:")
            for rank, (idx, _, final_balance) in enumerate(top_performers, start=1):
                # Intra-generational percentage gain
                previous_balance = env.balances[idx] if survival_streaks[idx] > 0 else env.initial_balance
                intra_gen_percentage_change = ((final_balance - previous_balance) / previous_balance) * 100 if previous_balance != 0 else 0
                # Total percentage gain
                total_percentage_change = ((final_balance - env.initial_balance) / env.initial_balance) * 100 if env.initial_balance != 0 else 0
                survival_streaks[idx] += 1
                print(f"Rank {rank}: Agent {idx + 1} with Balance = {final_balance:.2f}, Intra-Generational Percentage Change = {intra_gen_percentage_change:.2f}%, Total Percentage Change = {total_percentage_change:.2f}%")
            print(f"Total Profit: {total_profit:.2f}")
            print(f"Max Profit: {max_profit:.2f}")
            print(f"Min Profit: {min_profit:.2f}")
            print(f"Average Profit: {average_profit:.2f}")
            aggregate_profits.append(total_profit)
        else:
            print("No survivors in this generation.")
            aggregate_profits.append(0)

        
        

        with summary_writer.as_default():
            tf.summary.scalar('Total Profit', total_profit, step=generation+1)
            tf.summary.scalar('Max Profit', max_profit, step=generation+1)
            tf.summary.scalar('Min Profit', min_profit, step=generation+1)
            tf.summary.scalar('Average Profit', average_profit, step=generation+1)
            tf.summary.scalar('Number of Survivors', len(alive_agents), step=generation+1)

        if len(alive_agents) == 0:
            agents = [DQNAgent(state_size, action_size) for _ in range(population_size)]
            env.balances = [env.initial_balance] * population_size
        else:
            num_new_agents = population_size - len(alive_agents)
            new_agents = create_new_agents(alive_agents, num_new_agents, state_size, action_size, DQNAgent)
            agents = [agent for _, agent, _ in alive_agents] + new_agents
            # Only set initial balance for new agents
            env.balances = [final_balance if idx in [agent[0] for agent in alive_agents] else env.initial_balance for idx, final_balance in enumerate(env.balances)]

        for idx, agent in enumerate(tqdm(agents, desc="Saving Agents")):
            weights_file = os.path.join(weights_dir, f'agent_{idx}.weights.h5')
            agent.model.save_weights(weights_file)

    print("Training completed. Models saved.")

except KeyboardInterrupt:
    print("Training interrupted by user.")
    for idx, agent in enumerate(tqdm(agents, desc="Saving Agents")):
        weights_file = os.path.join(weights_dir, f'agent_{idx}.weights.h5')
        agent.model.save_weights(weights_file)

summary_writer.close()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(aggregate_profits) + 1), aggregate_profits, marker='o')
plt.title('Aggregate Profit Over Generations')
plt.xlabel('Generation')
plt.ylabel('Total Profit')
plt.grid(True)
plt.savefig('aggregate_profit.png')
plt.close()
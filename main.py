import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import pickle
import datetime
import os
import multiprocessing
import sys
import heapq  # Add this import

# Ensure the following code is only executed when the script is run directly
if __name__ == '__main__':
    # Set the multiprocessing start method (necessary for Windows)
    multiprocessing.set_start_method('spawn', force=True)

# Optional: Force TensorFlow to use CPU only (uncomment for testing purposes)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Limit GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print(f"Using GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Running on CPU.")

# Set TensorFlow logging level to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow messages

# Define the Baccarat Environment
class BaccaratEnv:
    def __init__(self, initial_balance=100):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shoe = self._generate_shoe()
        self.current_position = 0  # Tracks the current card position in the shoe
        self.cut_card_position = random.randint(60, 75)  # Cut card is placed towards the end
        self.allowed_bet_units = [1, 2, 5, 25, 100]

    def reset(self):
        """Resets the environment for a new episode."""
        self.balance = self.initial_balance
        self.shoe = self._generate_shoe()
        self.current_position = 0
        self.cut_card_position = random.randint(60, 75)
        # Burn a card at the beginning
        self._draw_card()
        return self._get_state()

    def _generate_shoe(self):
        """Generates a new shoe of 8 decks (416 cards)."""
        shoe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 8 * 4  # 8 decks of 52 cards
        random.shuffle(shoe)
        return shoe

    def _get_state(self):
        """Returns the current state of the game."""
        return [self.balance, len(self.shoe) - self.current_position]

    def _draw_card(self):
        """Draws a card from the shoe."""
        card = self.shoe[self.current_position]
        self.current_position += 1
        return card

    def _calculate_hand_value(self, cards):
        """Calculates the value of a hand in Baccarat."""
        value = sum(cards) % 10
        return value

    def step(self, action, bet_size):
        """Performs an action in the environment.

        action: 0 - Bet on Banker, 1 - Bet on Player, 2 - Bet on Tie
        bet_size: the amount to bet
        """
        if self.current_position >= self.cut_card_position or self.current_position >= len(self.shoe) - 6:
            # End of shoe
            return self._get_state(), 0, True, {}

        # Initial two cards for Player and Banker
        player_hand = [self._draw_card(), self._draw_card()]
        banker_hand = [self._draw_card(), self._draw_card()]

        # Calculate initial hand values
        player_value = self._calculate_hand_value(player_hand)
        banker_value = self._calculate_hand_value(banker_hand)

        # Natural win check
        if player_value >= 8 or banker_value >= 8:
            pass  # No further cards are drawn
        else:
            # Player's third card rule
            if player_value <= 5:
                player_hand.append(self._draw_card())
                player_value = self._calculate_hand_value(player_hand)

            # Banker's third card rule
            if banker_value <= 5:
                if len(player_hand) == 3:
                    player_third_card = player_hand[2]
                    if banker_value <= 2:
                        banker_hand.append(self._draw_card())
                    elif banker_value == 3 and player_third_card != 8:
                        banker_hand.append(self._draw_card())
                    elif banker_value == 4 and player_third_card in [2, 3, 4, 5, 6, 7]:
                        banker_hand.append(self._draw_card())
                    elif banker_value == 5 and player_third_card in [4, 5, 6, 7]:
                        banker_hand.append(self._draw_card())
                    elif banker_value == 6 and player_third_card in [6, 7]:
                        banker_hand.append(self._draw_card())
                else:
                    if banker_value <= 5:
                        banker_hand.append(self._draw_card())

        # Final hand values
        player_value = self._calculate_hand_value(player_hand)
        banker_value = self._calculate_hand_value(banker_hand)

        # Determine outcome
        if player_value > banker_value:
            outcome = 'Player'
        elif banker_value > player_value:
            outcome = 'Banker'
        else:
            outcome = 'Tie'

        # Determine reward
        if outcome == 'Tie' and action == 2:
            reward = bet_size * 8
        elif outcome == 'Tie':
            reward = -bet_size
        else:
            if action == 0 and outcome == 'Banker':
                reward = bet_size * 0.95  # 5% commission on Banker wins
            elif action == 1 and outcome == 'Player':
                reward = bet_size
            else:
                reward = -bet_size

        # Penalize the agent for losing balance
        if reward < 0:
            reward += -0.1 * abs(reward)  # Additional penalty for losing

        self.balance += reward
        done = self.balance < min(self.allowed_bet_units) or self.balance >= 10 * self.initial_balance or self.current_position >= self.cut_card_position or self.current_position >= len(self.shoe) - 6
        return self._get_state(), reward, done, {}

# Define the Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size, model=None):
        # Initialize the model
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action
            return np.random.choice(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough samples to replay
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(np.array([next_state]), verbose=0)[0]))
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically update the target model
        self.update_target_model()

# Mutation function for genetic algorithm
def mutate_weights(weights, mutation_rate=0.01):
    """Applies random noise to the model weights."""
    mutated_weights = []
    for w in weights:
        noise = np.random.randn(*w.shape) * mutation_rate
        mutated_weights.append(w + noise)
    return mutated_weights

# Main reinforcement learning loop
profits_per_episode = []
rewards_per_episode = []
state_size = 2  # Balance and remaining cards in shoe
action_size = 3  # Bet on Banker, Player, or Tie
population_size = 100  # Number of agents
episodes = 100  # Number of episodes per generation
generations = 25  # Number of generations

# Global dictionary to store agent models by index
agent_models = {}

# Initialize list to keep track of top agents across generations
top_agents = []  # List of tuples (-total_profit, idx, model_weights)
top_agents_size = 10  # Number of top agents to keep

# Function to train an agent (to be used in multiprocessing)
def train_agent(agent_info):
    idx, episodes, agent_model = agent_info
    # Each process must create its own environment and agent to avoid conflicts
    env = BaccaratEnv()
    state_size = len(env._get_state())
    action_size = 3  # Bet on Banker, Player, or Tie
    agent = DQNAgent(state_size, action_size, model=agent_model)

    # Do not load model weights; start from scratch if model is None
    if agent_model is None:
        print(f'Agent {idx + 1}: Starting training from scratch.')
    else:
        print(f'Agent {idx + 1}: Continuing training from previous generation.')

    total_profits = []
    total_rewards = []

    # Set up TensorBoard for this agent
    log_dir_agent = f"logs/fit/agent_{idx}"
    agent_summary_writer = tf.summary.create_file_writer(log_dir_agent)

    # For aggregate metrics
    total_profit_all = 0  # Total profit for this agent across all episodes

    for e in range(episodes):
        state = env.reset()
        done = False
        total_profit = 0
        total_reward = 0
        step_count = 0

        # Console logging at the start of each episode
        print(f"Agent {idx + 1} - Episode {e + 1}/{episodes} starting. Initial Balance: {env.balance}")

        while not done and env.balance >= min(env.allowed_bet_units):
            action = agent.act(state)
            bet_size = random.choice([unit for unit in env.allowed_bet_units if unit <= env.balance])
            next_state, reward, done, _ = env.step(action, bet_size)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_profit += reward
            total_reward += reward
            step_count += 1

            # Optional: Detailed step logging (commented out to reduce verbosity)
            # print(f"    Agent {idx + 1} - Step {step_count}: Action: {action}, Bet Size: {bet_size}, Reward: {reward:.2f}, New Balance: {env.balance:.2f}")

        if len(agent.memory) >= 256:
            agent.replay(256)

        # Log metrics for this episode
        with agent_summary_writer.as_default():
            tf.summary.scalar('Total Profit', total_profit, step=e)
            tf.summary.scalar('Total Reward', total_reward, step=e)
            tf.summary.scalar('Epsilon', agent.epsilon, step=e)
            tf.summary.scalar('Remaining Balance', env.balance, step=e)
            agent_summary_writer.flush()  # Flush the writer to disk for live updates

        # Save agent weights every 50 episodes (optional)
        if (e + 1) % 50 == 0:
            agent.update_target_model()  # Update target model
            # Optionally save weights
            # agent.model.save_weights(f'baccarat_dqn_weights_agent_{idx}.weights.h5')
            print(f"Agent {idx + 1} - Episode {e + 1}: Weights updated.")

        total_profits.append(total_profit)
        total_rewards.append(total_reward)
        total_profit_all += total_profit  # Accumulate total profit for this agent

        # Console logging at the end of each episode
        print(f"Agent {idx + 1} - Episode {e + 1} completed. Total Profit: {total_profit:.2f}, Remaining Balance: {env.balance:.2f}, Epsilon: {agent.epsilon:.4f}")

        # Check if agent is bankrupt
        if env.balance < min(env.allowed_bet_units):
            print(f"Agent {idx + 1} ran out of balance.")
            break  # Exit early if agent is bankrupt

    # Determine if agent is bankrupt and set model_weights accordingly
    if env.balance < min(env.allowed_bet_units):
        model_weights = None  # Agent is bankrupt
    else:
        model_weights = agent.model.get_weights()

    # Return the total profit for this agent and the trained model
    return idx, total_profits, total_rewards, total_profit_all, model_weights

# Prepare agent information for multiprocessing
agent_infos = []

# Initialize agents without loading any pre-trained models
for idx in range(population_size):
    model = None  # Start with a fresh model
    agent_infos.append((idx, episodes, model))

# Use multiprocessing Pool to train agents in parallel
if __name__ == '__main__':
    # Adjust the number of processes based on CPU cores
    num_processes = min(multiprocessing.cpu_count(), 12)  # Limit to 4 processes or number of CPU cores

    # Set up TensorBoard for aggregate metrics
    log_dir_aggregate = "logs/fit/aggregate"
    aggregate_summary_writer = tf.summary.create_file_writer(log_dir_aggregate)

    print(f"Training started with {population_size} agents, {episodes} episodes per agent.")

    # Initialize list to keep track of top agents across generations
    top_agents = []  # List of tuples (-total_profit, idx, model_weights)
    top_agents_size = 10  # Number of top agents to keep

    # Loop through generations
    for generation in range(generations):
        print(f"\n--- Starting Generation {generation + 1}/{generations} ---\n")
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Update agent_infos with the correct number of episodes for this generation
            agent_infos = []
            for idx in range(population_size):
                # Check if agent is still alive
                if idx in agent_models:
                    model_weights = agent_models[idx]
                    model = DQNAgent(state_size, action_size)._build_model()
                    model.set_weights(model_weights)
                    agent_infos.append((idx, episodes, model))
                else:
                    # Agent is dead; will be replaced later
                    agent_infos.append((idx, episodes, None))

            try:
                # Use imap_unordered to get an iterator
                results = pool.imap_unordered(train_agent, agent_infos)

                # Collect results as they become available
                results_list = []
                for result in results:
                    results_list.append(result)
            except KeyboardInterrupt:
                print("KeyboardInterrupt received. Terminating the pool.")
                pool.terminate()
                pool.join()
                sys.exit()
            except Exception as e:
                print(f"An error occurred: {e}")
                pool.terminate()
                pool.join()
                sys.exit()
            else:
                pool.close()
                pool.join()

        # Aggregate results
        total_profits_all_agents = []
        total_rewards_all_agents = []
        total_profit_all_agents = 0  # Total profit across all agents for this generation
        new_agent_models = {}
        alive_agents = 0

        # New list to keep track of agents' performance for this generation
        agents_performance = []

        for idx, total_profits, total_rewards, total_profit_all, model_weights in results_list:
            profits_per_episode.extend(total_profits)
            rewards_per_episode.extend(total_rewards)
            total_profits_all_agents.append(total_profit_all)
            total_profit_all_agents += total_profit_all

            # Record the agent's performance
            agents_performance.append((total_profit_all, idx, model_weights))

            if model_weights is not None:
                new_agent_models[idx] = model_weights  # Keep agent if not bankrupt
                alive_agents += 1
                print(f"Agent {idx + 1} survived Generation {generation + 1}. Total Profit: {total_profit_all:.2f}")
            else:
                print(f"Agent {idx + 1} did not survive Generation {generation + 1} (bankrupt).")

        # Update the top agents list
        # We use a heap to keep the top N agents
        for total_profit, idx, model_weights in agents_performance:
            if model_weights is not None:
                # Use negative total_profit because heapq is a min-heap
                heapq.heappush(top_agents, (-total_profit, idx, model_weights))
                if len(top_agents) > top_agents_size:
                    heapq.heappop(top_agents)

        agent_models = new_agent_models  # Update the agent models for the next generation

        # Print the total survivor count
        print(f"Total survivors in Generation {generation + 1}: {alive_agents}")

        # If agents have died, create new ones by mutating successful agents
        if alive_agents < population_size:
            num_new_agents = population_size - alive_agents
            print(f"Creating {num_new_agents} new agents by mutating successful agents.")

            # Get indices of successful agents
            successful_agent_indices = list(new_agent_models.keys())

            if not successful_agent_indices:
                # If no agents survived, use top agents from previous generations
                if top_agents:
                    print("No surviving agents. Using top agents from previous generations for mutation.")
                    # Extract model weights from top agents
                    successful_agents_for_mutation = [model_weights for _, _, model_weights in top_agents]
                else:
                    print("No top agents available. Re-initializing agents from scratch.")
                    # Re-initialize agents from scratch
                    new_agent_models = {}
                    for idx in range(population_size):
                        new_agent_models[idx] = DQNAgent(state_size, action_size)._build_model().get_weights()
                    agent_models = new_agent_models
                    continue  # Skip mutation step
            else:
                # Use surviving agents for mutation
                successful_agents_for_mutation = [new_agent_models[idx] for idx in successful_agent_indices]

            # Get the list of indices that are missing
            missing_indices = [idx for idx in range(population_size) if idx not in new_agent_models]
            for idx in missing_indices:
                # Select a parent agent
                parent_weights = random.choice(successful_agents_for_mutation)
                # Apply mutation
                mutated_weights = mutate_weights(parent_weights)
                new_agent_models[idx] = mutated_weights
                print(f"Created new agent {idx + 1} from mutation.")

        # Update agent_models with new_agent_models
        agent_models = new_agent_models

        # Log aggregate metrics for this generation
        with aggregate_summary_writer.as_default():
            for idx, total_profit in enumerate(total_profits_all_agents):
                tf.summary.scalar(f'Total Profit Agent {idx}', total_profit, step=generation)
            tf.summary.scalar('Total Profit All Agents', total_profit_all_agents, step=generation)
            aggregate_summary_writer.flush()  # Flush the writer to disk for live updates

        print(f"Generation {generation + 1} completed. Total Profit Across All Agents: {total_profit_all_agents:.2f}")

    # Save training metrics at the end of training
    with open('training_metrics.pkl', 'wb') as f:
        pickle.dump({'profits': profits_per_episode, 'rewards': rewards_per_episode}, f)

    # Plotting the results (optional)
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, len(profits_per_episode) + 1), profits_per_episode, label='Profit per Episode', color='b')
    # plt.plot(range(1, len(rewards_per_episode) + 1), rewards_per_episode, label='Total Reward per Episode', color='g')
    # plt.xlabel('Episode')
    # plt.ylabel('Amount')
    # plt.title('Reinforcement Learning in Baccarat Game - Profit and Reward per Episode')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

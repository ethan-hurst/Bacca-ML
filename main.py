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

# Define the Baccarat Environment
class BaccaratEnv:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shoe = self._generate_shoe()
        self.current_position = 0  # Tracks the current card position in the shoe
        self.cut_card_position = random.randint(60, 75)  # Cut card is placed towards the end
        self.allowed_bet_units = [1, 2, 5, 25, 100, 500, 1000]

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
            reward = bet_size if action in [0, 1] and outcome in ['Banker', 'Player'] else -bet_size

        # Penalize the agent for losing balance
        if reward < 0:
            reward += -0.1 * abs(reward)  # Additional penalty for losing

        self.balance += reward
        done = self.balance < min(self.allowed_bet_units) or self.balance >= 10 * self.initial_balance or self.current_position >= self.cut_card_position or self.current_position >= len(self.shoe) - 6
        return self._get_state(), reward, done, {}

# Define the Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.990
        self.learning_rate = 0.0001
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
            # Softmax exploration for better action diversity
            action_probs = np.ones(self.action_size) / self.action_size
            return np.random.choice(self.action_size, p=action_probs)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(np.array([next_state]))[0]))
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically update the target model
        if batch_size % 10 == 0:
            self.update_target_model()

# Set up TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)

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

# Main reinforcement learning loop
profits_per_episode = []
rewards_per_episode = []
env = BaccaratEnv()
state_size = len(env._get_state())
action_size = 3  # Bet on Banker, Player, or Tie
population_size = 100  # Set to 100 agents
agents = [DQNAgent(state_size, action_size) for _ in range(population_size)]
episodes = 500  # Reduced to 500 episodes for generational improvement

# Load model weights if available before starting training
from multiprocessing import Pool

def train_agent(agent_info):
            idx, agent, episodes = agent_info
            state = env.reset()
            done = False
            total_profit = 0
            total_reward = 0
            while not done and env.balance >= min(env.allowed_bet_units):
                action = agent.act(state)
                bet_size = random.choice([unit for unit in env.allowed_bet_units if unit <= env.balance])
                next_state, reward, done, _ = env.step(action, bet_size)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_profit += reward
                total_reward += reward
                print(f"    Step: Agent {idx + 1}, Action: {action}, Bet Size: {bet_size}, Reward: {reward}, Balance: {env.balance}")
            if len(agent.memory) > 64:
                agent.replay(256)
            return idx, total_profit, total_reward
with Pool(processes=4) as pool:
    results = pool.map(train_agent, [(idx, agent, episodes) for idx, agent in enumerate(agents)])


for idx, total_profit, total_reward in results:
    try:
        agent.model.load_weights(f'baccarat_dqn_weights_agent_{idx}.weights.h5')
        print(f'Loaded saved weights for agent {idx} successfully.')
    except FileNotFoundError:
        print(f'No saved weights found for agent {idx}, starting fresh.')

for generation in range(5):  # Run for 5 generations
    print(f"Starting generation {generation + 1}")
    for e in range(episodes):
        print(f"Starting episode {e + 1} of generation {generation + 1}")
        total_generation_profit = 0  # Track total profit for all agents in this episode
        total_generation_reward = 0  # Track total reward for all agents in this episode
        num_active_agents = 0  # Track the number of agents that are still active

        for idx, agent in enumerate(agents):
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            state = env.reset()
            done = False
            total_profit = 0
            total_reward = 0

            while not done and env.balance >= min(env.allowed_bet_units):
                if env.balance < min(env.allowed_bet_units):
                    print(f"Agent {idx + 1} has died (balance below minimum bet unit). Agents left: {num_active_agents - 1}")
                action = agent.act(state)
                bet_size = random.choice([unit for unit in env.allowed_bet_units if unit <= env.balance])
                next_state, reward, done, _ = env.step(action, bet_size)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_profit += reward
                total_reward += reward

            # Replay the experience with a batch size of 32
            if len(agent.memory) > 64:
                agent.replay(256)

            # Accumulate the profit and reward of each agent for this episode
            total_generation_profit += total_profit
            total_generation_reward += total_reward

            # Increment the count of active agents
            if env.balance >= min(env.allowed_bet_units):
                num_active_agents += 1
            print(f"Agent {idx + 1} completed. Total Profit: {total_profit}, Total Reward: {total_reward}, Remaining Balance: {env.balance}")

            # Save metrics for individual agents (optional)
            with summary_writer.as_default():
                tf.summary.scalar(f'Total Profit Agent {idx}', total_profit, step=e + (generation * episodes))
                tf.summary.scalar(f'Total Reward Agent {idx}', total_reward, step=e + (generation * episodes))
                tf.summary.scalar(f'Epsilon Agent {idx}', agent.epsilon, step=e + (generation * episodes))

            # Save agent weights every 50 episodes
            if (e + 1) % 50 == 0:
                agent.update_target_model()  # Update target model
                agent.model.save_weights(f'baccarat_dqn_weights_agent_{idx}.weights.h5')

        # Log generation metrics
        profits_per_episode.append(total_generation_profit)
        rewards_per_episode.append(total_generation_reward)
        print(f"Total Generation Profit: {total_generation_profit}, Total Generation Reward: {total_generation_reward}, Active Agents: {num_active_agents}")

        with summary_writer.as_default():
            tf.summary.scalar('Total Generation Profit', total_generation_profit, step=e + (generation * episodes))
            tf.summary.scalar('Total Generation Reward', total_generation_reward, step=e + (generation * episodes))

# Save training metrics at the end of training
with open('training_metrics.pkl', 'wb') as f:
    pickle.dump({'profits': profits_per_episode, 'rewards': rewards_per_episode}, f)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(profits_per_episode) + 1), profits_per_episode, label='Profit per Episode', color='b')
plt.plot(range(1, len(rewards_per_episode) + 1), rewards_per_episode, label='Total Reward per Episode', color='g')
plt.xlabel('Episode')
plt.ylabel('Amount')
plt.title('Reinforcement Learning in Baccarat Game - Profit and Reward per Episode')
plt.legend()
plt.grid(True)
plt.show()
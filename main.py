import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque

# Define the Baccarat Environment
class BaccaratEnv:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shoe = self._generate_shoe()
        self.current_position = 0  # Tracks the current card position in the shoe
        self.cut_card_position = random.randint(60, 75)  # Cut card is placed towards the end

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
        reward = 0
        if action == 0 and outcome == 'Banker':
            reward = bet_size * 0.95
        elif action == 1 and outcome == 'Player':
            reward = bet_size
        elif action == 2 and outcome == 'Tie':
            reward = bet_size * 8
        else:
            reward = -bet_size

        self.balance += reward
        done = self.balance <= 0 or self.balance >= 10 * self.initial_balance or self.current_position >= self.cut_card_position or self.current_position >= len(self.shoe) - 6
        return self._get_state(), reward, done, {}

# Define the Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.990
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0]))
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main reinforcement learning loop
profits_per_episode = []
rewards_per_episode = []
env = BaccaratEnv()
state_size = len(env._get_state())
action_size = 3  # Bet on Banker, Player, or Tie
agent = DQNAgent(state_size, action_size)
population_size = 50
episodes = 5000

for e in range(episodes):
    state = env.reset()
    done = False
    total_profit = 0
    total_reward = 0
    while not done and env.balance < 10 * env.initial_balance:
        action = agent.act(state)
        bet_size = random.randint(1, 100)  # For simplicity, use a random bet size
        next_state, reward, done, _ = env.step(action, bet_size)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_profit += reward
        total_reward += reward

    # Replay the experience with a batch size of 32
    if len(agent.memory) > 32:
        agent.replay(32)

    profits_per_episode.append(total_profit)
    rewards_per_episode.append(total_reward)
    print(f"Episode {e + 1}/{episodes}, Total Profit: {total_profit}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2}")

# Plotting the results
profits = [random.randint(-100, 100) for _ in range(episodes)]  # Replace with actual profit values
plt.figure(figsize=(12, 6))
plt.plot(range(1, episodes + 1), profits_per_episode, label='Profit per Episode', color='b')
plt.plot(range(1, episodes + 1), rewards_per_episode, label='Total Reward per Episode', color='g')
plt.xlabel('Episode')
plt.ylabel('Amount')
plt.title('Reinforcement Learning in Baccarat Game - Profit and Reward per Episode')
plt.legend()
plt.grid(True)
plt.show()
plt.xlabel('Episode')
plt.ylabel('Profit')
plt.title('Reinforcement Learning in Baccarat Game')
plt.legend()
plt.show()
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import os

# Define the Baccarat Environment
class BaccaratEnv:
    def __init__(self, num_agents, initial_balance=100, history_length=100):
        self.num_agents = num_agents
        self.initial_balance = initial_balance
        self.balances = [initial_balance] * num_agents  # Individual balances for each agent
        self.shoe = self._generate_shoe()
        self.current_position = 0
        self.cut_card_position = random.randint(60, 75)
        self.allowed_bet_units = [1, 2, 5, 25, 100]
        self.history_length = history_length
        self.outcome_history = []  # Stores the history of outcomes

    def reset(self):
        """Resets the environment for a new shoe."""
        self.shoe = self._generate_shoe()
        self.current_position = 0
        self.cut_card_position = random.randint(60, 75)
        self.outcome_history = []
        # Burn a card at the beginning
        self._draw_card()
        # Reset balances
        self.balances = [self.initial_balance] * self.num_agents
        return self._get_states()

    def _generate_shoe(self):
        """Generates a new shoe of 8 decks (416 cards)."""
        shoe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 8 * 4  # 8 decks of 52 cards
        random.shuffle(shoe)
        return shoe

    def _get_states(self):
        """Returns the current state for all agents."""
        states = []
        for balance in self.balances:
            state = [balance, len(self.shoe) - self.current_position] + self.outcome_history[-self.history_length:]
            # Pad the outcome history if it's shorter than history_length
            if len(state) < 2 + self.history_length:
                padding = [0] * (2 + self.history_length - len(state))
                state += padding
            states.append(state)
        return states

    def _draw_card(self):
        """Draws a card from the shoe."""
        card = self.shoe[self.current_position]
        self.current_position += 1
        return card

    def _calculate_hand_value(self, cards):
        """Calculates the value of a hand in Baccarat."""
        value = sum(cards) % 10
        return value

    def step(self, actions, bet_sizes):
        """Processes a single hand with bets from all agents.

        actions: list of actions from all agents
        bet_sizes: list of bet sizes from all agents
        """
        if self.current_position >= self.cut_card_position or self.current_position >= len(self.shoe) - 6:
            # End of shoe
            done = True
            return self._get_states(), [0]*self.num_agents, done, {}
        else:
            done = False

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

        # Update outcome history
        outcome_encoding = {'Player': 0, 'Banker': 1, 'Tie': 2}
        self.outcome_history.append(outcome_encoding[outcome])

        # Distribute rewards and update balances
        rewards = []
        for idx, (action, bet_size) in enumerate(zip(actions, bet_sizes)):
            if self.balances[idx] < bet_size or bet_size == 0:
                # Agent cannot bet; skip
                rewards.append(0)
                continue

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

            self.balances[idx] += reward
            rewards.append(reward)

        return self._get_states(), rewards, done, {}

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
state_size = 2 + 100  # Balance, remaining cards, and outcome history length
action_size = 3  # Bet on Banker, Player, or Tie
population_size = 100  # Number of agents
generations = 25  # Number of generations

# Initialize agents
agents = []
for idx in range(population_size):
    agent = DQNAgent(state_size, action_size)
    weights_file = f'agent_{idx}_weights.h5'
    if os.path.exists(weights_file):
        agent.model.load_weights(weights_file)
        print(f"Loaded weights for agent {idx + 1}")
    else:
        print(f"No saved weights for agent {idx + 1}, initializing new agent.")
    agents.append(agent)

# Initialize the shared environment
env = BaccaratEnv(num_agents=population_size, history_length=100)

# Training loop
for generation in range(generations):
    print(f"\n--- Starting Generation {generation + 1}/{generations} ---\n")
    
    # Reset environment for new shoe
    states = env.reset()
    
    done = False
    step_count = 0
    while not done:
        actions = []
        bet_sizes = []
        for idx, agent in enumerate(agents):
            state = states[idx]
            if env.balances[idx] >= min(env.allowed_bet_units):
                action = agent.act(state)
                bet_size = random.choice([unit for unit in env.allowed_bet_units if unit <= env.balances[idx]])
            else:
                action = 0  # Default action
                bet_size = 0
            actions.append(action)
            bet_sizes.append(bet_size)
        
        # Step the environment
        next_states, rewards, done, _ = env.step(actions, bet_sizes)
        
        # Agents remember experiences and potentially learn
        for idx, agent in enumerate(agents):
            state = states[idx]
            action = actions[idx]
            reward = rewards[idx]
            next_state = next_states[idx]
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) >= 256:
                agent.replay(256)
        
        states = next_states
        step_count += 1

    # After each shoe (episode), handle agent survival and mutation
    alive_agents = []
    dead_agents_indices = []
    survivor_profits = []

    for idx, agent in enumerate(agents):
        final_balance = env.balances[idx]
        initial_balance = env.initial_balance
        profit = final_balance - initial_balance
        if final_balance >= min(env.allowed_bet_units):
            alive_agents.append((idx, agent, final_balance))
            survivor_profits.append(profit)
        else:
            dead_agents_indices.append(idx)
            print(f"Agent {idx + 1} went bankrupt.")

    # Log the number of survivors
    print(f"Total survivors in Generation {generation + 1}: {len(alive_agents)}")

    # Display profits
    if survivor_profits:
        total_profit = sum(survivor_profits)
        max_profit = max(survivor_profits)
        min_profit = min(survivor_profits)
        print(f"Generation {generation + 1} Survivors' Profits:")
        print(f"Total Profit: {total_profit:.2f}")
        print(f"Max Profit: {max_profit:.2f}")
        print(f"Min Profit: {min_profit:.2f}")
    else:
        print("No survivors in this generation.")

    # Handle agent replacement
    if len(alive_agents) == 0:
        print("All agents went bankrupt. Reinitializing agents.")
        agents = []
        for idx in range(population_size):
            agent = DQNAgent(state_size, action_size)
            agents.append(agent)
    else:
        # Create new agents by mutating survivors
        num_new_agents = population_size - len(alive_agents)
        print(f"Creating {num_new_agents} new agents by mutating survivors.")
        new_agents = []
        for _ in range(num_new_agents):
            parent_idx, parent_agent, _ = random.choice(alive_agents)
            mutated_weights = mutate_weights(parent_agent.model.get_weights())
            new_agent = DQNAgent(state_size, action_size)
            new_agent.model.set_weights(mutated_weights)
            new_agents.append(new_agent)
        # Update agents list
        agents = [agent for _, agent, _ in alive_agents] + new_agents
        # Reset balances for all agents
        env.balances = [env.initial_balance] * population_size

    # Save models after each generation
    for idx, agent in enumerate(agents):
        agent.model.save_weights(f'agent_{idx}_weights.h5')
        print(f"Saved weights for agent {idx + 1}")

print("Training completed. Models saved.")

# dqn_agent.py
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, BatchNormalization, Dropout
import tensorflow.keras.backend as K
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Use deque for efficient memory management
        self.priorities = deque(maxlen=10000)
        self.gamma = 0.99  # Discount rate
        self.epsilon = 20  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Adjusted for better exploration
        self.learning_rate = 0.00025  # Adjusted learning rate
        self.learning_rate_advantage = 0.0005  # Separate learning rate for advantage
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.update_target_counter = 0  # For updating target model periodically

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Dueling architecture
        value_fc = Dense(256, activation='relu')(x)
        value = Dense(1, activation='linear')(value_fc)

        advantage_fc = Dense(256, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear')(advantage_fc)

        def combine_value_and_advantage(args):
            value, advantage = args
            return value + (advantage - K.mean(advantage, axis=1, keepdims=True))

        q_values = Lambda(combine_value_and_advantage, name='Q_Values')([value, advantage])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32).reshape(1, self.state_size)
        next_state = np.array(next_state, dtype=np.float32).reshape(1, self.state_size)
        target = reward
        if not done:
            next_action = np.argmax(self.model(next_state, training=False)[0])
            target += self.gamma * self.target_model(next_state, training=False)[0][next_action]
        current_q = self.model(state, training=False)[0][action]
        td_error = abs(target - current_q)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(td_error)

    def act(self, state, balance, sit_out_count):
        state = np.array(state, dtype=np.float32).reshape(1, self.state_size)

        # Adaptive exploration: Reduce epsilon for high balance, increase for low balance
        if balance > 200:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.95)
        elif balance < 50:
            self.epsilon = min(20, self.epsilon * 1.05)

        # If the agent has sat out for 6 hands, force a minimum bet
        if sit_out_count >= 6:
            return np.random.choice(self.action_size), min(self.action_size)

        # Hail Mary strategy for low balance (below 20)
        if balance < 20:
            bet_size = balance * random.uniform(0.1, 0.5)  # Proportional bet for Hail Mary
            return np.random.choice(self.action_size), bet_size

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size), random.choice([unit for unit in [1, 2, 5, 25, 100] if unit <= balance])

        act_values = self.model(state, training=False)
        return np.argmax(act_values[0]), random.choice([unit for unit in [1, 2, 5, 25, 100] if unit <= balance])

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        memory_length = len(self.memory)
        if memory_length < batch_size:
            return
        
        # Prioritized experience replay with scaling
        scaled_priorities = np.array(self.priorities) ** 0.6
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(memory_length, batch_size, p=sample_probabilities)
        minibatch = [self.memory[i] for i in indices]

        states = np.array([data[0].flatten() for data in minibatch], dtype=np.float32)
        actions = np.array([data[1] for data in minibatch])
        rewards = np.array([data[2] for data in minibatch])
        next_states = np.array([data[3].flatten() for data in minibatch], dtype=np.float32)
        dones = np.array([data[4] for data in minibatch])

        # Double Q-Learning update
        target = self.model(states, training=False).numpy()
        target_next = self.model(next_states, training=False).numpy()
        target_val = self.target_model(next_states, training=False).numpy()

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + self.gamma * target_val[i][a]

        # Update model with gradient clipping
        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_counter += 1
        if self.update_target_counter % 1000 == 0:
            self.update_target_model()
# baccarat_env.py
import random
import numpy as np

class BaccaEnv:
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
        self.sit_out_counts = [0] * num_agents  # Track the number of consecutive sit-outs

    def reset(self, reset_balances=True):
        """Resets the environment for a new shoe."""
        self.shoe = self._generate_shoe()
        self.current_position = 0
        self.cut_card_position = random.randint(60, 75)
        self.outcome_history = []
        self.sit_out_counts = [0] * self.num_agents  # Reset sit-out counts per shoe
        self._draw_card()  # Burn a card at the beginning
        if reset_balances:
            self.balances = [self.initial_balance] * self.num_agents
        return self._get_states()

    def _generate_shoe(self):
        """Generates a new shoe of 8 decks (416 cards)."""
        shoe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 8 * 4  # 8 decks of 52 cards
        random.shuffle(shoe)
        return shoe

    def _draw_card(self):
        """Draws a card from the shoe."""
        card = self.shoe[self.current_position]
        self.current_position += 1
        return card

    def _calculate_hand_value(self, cards):
        """Calculates the value of a hand in Baccarat."""
        value = sum(cards) % 10
        return value

    def _normalize_state(self, state):
        state[0] = float(state[0]) / self.initial_balance
        state[1] = float(state[1]) / len(self.shoe)
        return state

    def _get_states(self):
        states = []
        for balance in self.balances:
            state = [balance, len(self.shoe) - self.current_position] + self.outcome_history[-self.history_length:]
            if len(state) > 2 + self.history_length:
                state = state[:2 + self.history_length]
            else:
                padding = [0] * (2 + self.history_length - len(state))
                state += padding
            state = self._normalize_state(state)
            state = np.array(state, dtype=np.float32)
            states.append(state)
        return states

    def step(self, actions, bet_sizes):
        """Processes a single hand with bets from all agents."""
        if self.current_position >= self.cut_card_position or self.current_position >= len(self.shoe) - 6:
            done = True
            return self._get_states(), [0] * self.num_agents, done, {}
        else:
            done = False

        player_hand = [self._draw_card(), self._draw_card()]
        banker_hand = [self._draw_card(), self._draw_card()]

        player_value = self._calculate_hand_value(player_hand)
        banker_value = self._calculate_hand_value(banker_hand)

        if player_value >= 8 or banker_value >= 8:
            pass
        else:
            if player_value <= 5:
                player_hand.append(self._draw_card())
                player_value = self._calculate_hand_value(player_hand)
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

        player_value = self._calculate_hand_value(player_hand)
        banker_value = self._calculate_hand_value(banker_hand)

        if player_value > banker_value:
            outcome = 'Player'
        elif banker_value > player_value:
            outcome = 'Banker'
        else:
            outcome = 'Tie'

        outcome_encoding = {'Player': 0, 'Banker': 1, 'Tie': 2}
        self.outcome_history.append(outcome_encoding[outcome])

        rewards = []
        for idx, (action, bet_size) in enumerate(zip(actions, bet_sizes)):
            if self.balances[idx] < bet_size or bet_size == 0:
                rewards.append(-1)
                self.sit_out_counts[idx] += 1
                continue

            reward = 0
            if outcome == 'Tie' and action == 2:
                reward = bet_size * 8
            elif outcome == 'Tie':
                reward = -bet_size
            else:
                if action == 0 and outcome == 'Banker':
                    reward = bet_size * 0.95
                elif action == 1 and outcome == 'Player':
                    reward = bet_size
                else:
                    reward = -bet_size

            if action == 2:
                reward -= bet_size * 0.1

            self.balances[idx] += reward
            rewards.append(reward)
            self.sit_out_counts[idx] = 0

        return self._get_states(), rewards, done, {}
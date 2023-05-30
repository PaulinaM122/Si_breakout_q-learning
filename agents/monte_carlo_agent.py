import ast
import json
import re
import os
import random

import utilities
from agents.q_learning_agent import QLearningAgent
from globals.constants import *
import numpy as np
import matplotlib.pyplot as plt
from globals.direction import Direction as Dir


class MonteCarloAgent(QLearningAgent):
    def __init__(self):
        self.returns = {}  # słownik przechowujący sumy zwrotów dla każdego stanu i akcji
        self.state_action_counts = {}
        self.q_value_history = {
            Dir.LEFT: {},
            Dir.RIGHT: {},
            Dir.STAY: {}
        }
        super().__init__(q_values_file_name='q_values_monte.txt', screen_width=SCREEN_WIDTH_BIG)

    def get_state(self, ball, paddle, bricks):
        relative_position = [ball.get_relative_position(paddle).value]
        direction = ball.get_direction().value
        state_array = np.array(relative_position + [direction])
        return state_array

    def get_q_value(self, state, action):
        # funkcja zwracająca wartość Q-funkcji dla danego stanu i akcji
        # jeśli wartość nie istnieje, zwraca 0
        state_tuple = tuple(state)
        return self.q_values.get((state_tuple, action), 0)

    def update_q_value(self, episode):
        # funkcja aktualizująca wartość Q-funkcji na podstawie epizodu
        G = 0
        states_actions_visited = set()

        for state, action, reward in reversed(episode):
            state_tuple = tuple(state)

            if (state_tuple, action) not in states_actions_visited:
                states_actions_visited.add((state_tuple, action))

                self.returns[(state_tuple, action)] = self.returns.get((state_tuple, action), 0) + G
                self.state_action_counts[(state_tuple, action)] = self.state_action_counts.get((state_tuple, action), 0) + 1
                self.q_values[(state_tuple, action)] = self.returns[(state_tuple, action)] / self.state_action_counts[(state_tuple, action)]

                # Dodaj wartość Q-funkcji do historii
                if state_tuple not in self.q_value_history[action]:
                    self.q_value_history[action][state_tuple] = []
                self.q_value_history[action][state_tuple].append(self.q_values[(state_tuple, action)])

            G = self.gamma * G + reward

    def get_best_action(self, state, paddle):
        # funkcja zwracająca najlepszą akcję dla danego stanu
        possible_actions = self.get_possible_actions(paddle)
        state_tuple = tuple(state)
        max_value = max([self.get_q_value(state_tuple, action) for action in possible_actions])
        max_actions = [action for action in possible_actions if self.get_q_value(state_tuple, action) == max_value]
        if len(max_actions) > 1:
            return random.choice(max_actions)
        else:
            return max_actions[0]

    def get_action(self, state, paddle):
        # funkcja zwracająca akcję dla danego stanu
        # zgodnie z polityką epsilon-zachłanną
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(paddle))
        else:
            return self.get_best_action(state, paddle)

    def reset_episode_data(self):
        self.returns = {}
        self.state_action_counts = {}

    def load_q_values(self):
        file_path = './database_files/' + self.q_values_file_name
        self.q_values = {}

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)

            for key, value in data.items():
                # Parsowanie klucza w formacie "((3, 7), <Direction.RIGHT: 3>)"
                match = re.search(r'\(\((\d+), (\d+)\), <Direction\.[A-Z]+: (\d+)>', key)
                state1 = int(match.group(1))
                state2 = int(match.group(2))
                action = Dir(int(match.group(3)))

                # Parsowanie wartości jako float
                q_value = float(value)
                state = (state1,state2)
                # Przypisanie wartości do słownika q_values
                self.q_values[(state, action)] = q_value

        return self.q_values

    def plot_q_value_history(self):
        for action, state_values in self.q_value_history.items():
            plt.figure()
            for state, q_values in state_values.items():
                plt.plot(range(len(q_values)), q_values, label=str(state))
            plt.xlabel('Episodes')
            plt.ylabel('Q-value')
            plt.legend()
            plt.title(f'Action: {action.name}')
        plt.show()

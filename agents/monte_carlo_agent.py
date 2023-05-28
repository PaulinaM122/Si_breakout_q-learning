import json
import os
import random
from agents.q_learning_agent import QLearningAgent
from globals.constants import *
import numpy as np
import pickle


class MonteCarloAgent(QLearningAgent):
    def __init__(self):
        self.returns = {}  # słownik przechowujący sumy zwrotów dla każdego stanu i akcji
        self.state_action_counts = {}  # słownik przechowujący liczbę wystąpień danego stanu i akcji
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

    def save_q_values(self):
        # funkcja zapisująca wartości wytrenowanych q_values do pliku
        file_path = './database_files/' + self.q_values_file_name
        q_values_str = {str(key): value for key, value in self.q_values.items()}
        with open(file_path, 'w') as file:
            json.dump(q_values_str, file)

    def load_q_values(self):
        # funkcja wczytująca wartości wytrenowanych q_values z pliku
        file_path = './database_files/' + self.q_values_file_name
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as file:
                self.q_values = json.load(file)

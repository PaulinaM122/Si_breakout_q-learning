import abc
import random
import os
import re
import json
import utilities
from globals.direction import Direction as Dir
from agents.agent import Agent
from globals.constants import *


class QLearningAgent(Agent):

    def __init__(self, q_values_file_name, screen_width=SCREEN_WIDTH_SMALL, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # współczynnik uczenia się
        self.gamma = gamma  # współczynnik dyskontowania
        self.epsilon = epsilon  # współczynnik eksploracji
        self.q_values = {}  # słownik przechowujący wartości Q-funkcji dla każdego stanu i akcji
        self.screen_width = screen_width
        self.q_values_file_name = q_values_file_name
        super().__init__()

    @abc.abstractmethod
    def get_state(self, ball, paddle, bricks):
        pass

    def get_q_value(self, state, action):
        # funkcja zwracająca wartość Q-funkcji dla danego stanu i akcji
        # jeśli wartość nie istnieje, zwraca 0
        return self.q_values.get((state, action), 0)

    def update_q_value(self, state, action, next_state, reward, paddle):
        # funkcja aktualizująca wartość Q-funkcji dla danego stanu i akcji
        current_q = self.get_q_value(state, action)
        # uproszczone równanie Bellmana (można inne wzory na update q-value) :
        max_next_q = max(
            [self.get_q_value(next_state, next_action) for next_action in self.get_possible_actions(paddle)])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[(state, action)] = new_q

    def get_possible_actions(self, paddle):
        # funkcja zwracająca możliwe akcje dla danego stanu
        # paletka nie może wychodzić poza obszar gry
        paddle_position = paddle.pos()
        left_side = -self.screen_width / 2 + HALF_PADDLE
        right_side = self.screen_width / 2 - HALF_PADDLE
        if left_side < paddle_position[0] < right_side:
            return [Dir.LEFT, Dir.RIGHT, Dir.STAY]
        elif paddle_position[0] <= left_side:
            return [Dir.RIGHT, Dir.STAY]
        else:
            return [Dir.LEFT, Dir.STAY]

    def get_best_action(self, state, paddle):
        # funkcja zwracająca najlepszą akcję dla danego stanu
        possible_actions = self.get_possible_actions(paddle)
        max_value = max([self.get_q_value(state, action) for action in possible_actions])
        max_actions = [action for action in possible_actions if self.get_q_value(state, action) == max_value]
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

    def evaluate(self):
        success_rate = sum(self.success_history) / (self.num_games * 3)
        avg_reward = sum(self.reward_history) / self.num_games
        return success_rate, avg_reward

    def save_q_values(self):
        # funkcja zapisująca wartości wytrenowanych q_values do pliku
        file_path = './database_files/' + self.q_values_file_name
        with open(file_path, 'w') as file:
            file.write(json.dumps(utilities.map_dict_to_str(self.q_values), indent=0))

    def load_q_values2(self):
        file_path = './database_files/' + self.q_values_file_name
        self.q_values = {}

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.decoder.JSONDecodeError:
                    return self.q_values

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

    def load_q_values(self):
        file_path = './database_files/' + self.q_values_file_name
        self.q_values = {}

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.decoder.JSONDecodeError:
                    return self.q_values

            for key, value in data.items():
                # Parsowanie klucza w formacie "((<Direction.STAY: 1>, <Direction.UP_RIGHT: 4>), <Direction.STAY: 1>)"
                match = re.search(
                    r'\(\(<Direction\.[A-Z_]+: (\d+)>, <Direction\.[A-Z_]+: (\d+)>\), <Direction\.[A-Z_]+: (\d+)>', key)
                state1 = Dir(int(eval(match.group(1))))
                state2 = Dir(int(eval(match.group(2))))
                action = Dir(int(eval(match.group(3))))

                # Parsowanie wartości jako float
                q_value = float(value)
                state = (state1, state2)
                # Przypisanie wartości do słownika q_values
                self.q_values[(state, action)] = q_value

        return self.q_values

import random
import json
import utilities
import os
from agent import Agent
from Direction import Direction as Dir


class QLearningSimpleAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # współczynnik uczenia się
        self.gamma = gamma  # współczynnik dyskontowania
        self.epsilon = epsilon  # współczynnik eksploracji
        self.q_values = {}  # słownik przechowujący wartości Q-funkcji dla każdego stanu i akcji
        super().__init__()

    def get_state(self, ball, paddle, bricks):
        return ball.get_relative_position(paddle), ball.get_direction()
        # return ball.get_relative_position(paddle), ball.get_direction(), ball.get_distance()

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
        screen_width = 1200
        left_side = -1 * (screen_width / 2) + 100
        right_side = screen_width / 2 - 100
        if paddle_position[0] > left_side and paddle_position[0] < right_side:
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
        # TODO: możemy tu dać np obliczanie średnich, żeby zobaczyć, jak zmienić parametry w treningu i na koniec do tworzenia staystyk/wykresów
        success_rate = sum(self.success_history) / (self.num_games * 3)
        avg_reward = sum(self.reward_history) / self.num_games
        return success_rate, avg_reward

    def save_q_values(self):
        # funkcja zapisująca wartości wytrenowanych q_values do pliku q_values.txt
        with open('q_values_simple.txt', 'w') as file:
            file.write(json.dumps(utilities.map_dict_to_str(self.q_values), indent=0))

    def load_q_values(self):
        # funkcja wczytująca wartości wytrenowanych q_values z pliku q_values.txt
        if os.path.getsize('q_values_simple.txt') > 0:
            with open('q_values_simple.txt', 'r') as file:
                q_values_str = file.read()
                self.q_values = utilities.map_str_to_dict(json.loads(q_values_str))

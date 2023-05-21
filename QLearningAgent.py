import random
import json
import utilities
import os
from agent import Agent
from Direction import Direction as Dir


class QLearningAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # współczynnik uczenia się
        self.gamma = gamma  # współczynnik dyskontowania
        self.epsilon = epsilon  # współczynnik eksploracji
        self.q_values = {}  # słownik przechowujący wartości Q-funkcji dla każdego stanu i akcji
        super().__init__()

    def get_state(self, ball, paddle, bricks):
        return ball.pos(), paddle.pos(), bricks.get_state()

    def get_q_value(self, state, action):
        # funkcja zwracająca wartość Q-funkcji dla danego stanu i akcji
        # jeśli wartość nie istnieje, zwraca 0
        return self.q_values.get((state, action), 0)

    def update_q_value(self, state, action, next_state, reward):
        # funkcja aktualizująca wartość Q-funkcji dla danego stanu i akcji
        current_q = self.get_q_value(state, action)
        # uproszczone równanie Bellmana (można inne wzory na update q-value) :
        max_next_q = max(
            [self.get_q_value(next_state, next_action) for next_action in self.get_possible_actions(next_state)])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[(state, action)] = new_q


    def get_possible_actions(self, state):
        # funkcja zwracająca możliwe akcje dla danego stanu
        # paletka nie może wychodzić poza obszar gry
        paddle_position = state[1]
        if (-210 < paddle_position[0] < 210):
            return [Dir.LEFT, Dir.RIGHT, Dir.STAY]
        elif paddle_position[0] <= -210:
            return [Dir.RIGHT, Dir.STAY]
        else:
            return [Dir.LEFT, Dir.STAY]

    def get_best_action(self, state):
        # funkcja zwracająca najlepszą akcję dla danego stanu
        possible_actions = self.get_possible_actions(state)

        if not possible_actions:
            return None

        # Sprawdź czy wszystkie wartości dla danego stanu są równe 0
        all_zero_values = all(self.get_q_value(state, action) == 0 for action in possible_actions)

        if all_zero_values:
            # Jeśli wszystkie wartości są równe 0, wylosuj jeden ruch
            return random.choice(possible_actions)
        else:
            return max(possible_actions, key=lambda action: self.get_q_value(state, action))

    def get_action(self, state):
        # funkcja zwracająca akcję dla danego stanu
        # zgodnie z polityką epsilon-zachłanną
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(state))
        else:
            return self.get_best_action(state)

    def evaluate(self):
        # TODO: możemy tu dać np obliczanie średnich, żeby zobaczyć, jak zmienić parametry w treningu i na koniec do tworzenia staystyk/wykresów
        success_rate = sum(self.success_history) / (self.num_games * 3)
        avg_reward = sum(self.reward_history) / self.num_games
        return success_rate, avg_reward

    def save_q_values(self):
        # funkcja zapisująca wartości wytrenowanych q_values do pliku q_values.txt
        with open('q_values.txt', 'w') as file:
            file.write(json.dumps(utilities.map_dict_to_str(self.q_values), indent=0))

    def load_q_values(self):
        # funkcja wczytująca wartości wytrenowanych q_values z pliku q_values.txt
        if os.path.getsize('q_values.txt') > 0:
            with open('q_values.txt', 'r') as file:
                q_values_str = file.read()
                self.q_values = utilities.map_str_to_dict(json.loads(q_values_str))


    def change_epsilon(self):
        self.epsilon = 0.9

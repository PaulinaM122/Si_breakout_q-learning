import random
from Direction import Direction as Dir


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # współczynnik uczenia się
        self.gamma = gamma  # współczynnik dyskontowania
        self.epsilon = epsilon  # współczynnik eksploracji
        self.q_values = {}  # słownik przechowujący wartości Q-funkcji dla każdego stanu i akcji
        self.num_games = 0  # licznik gier
        self.max_num_games = 10000  # maksymalna liczba gier do rozegrania

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
        if paddle_position[0] > -420 and paddle_position[0] < 420:
            return [Dir.LEFT, Dir.RIGHT, Dir.STAY]
        elif paddle_position[0] <= -420:
            return [Dir.RIGHT, Dir.STAY]
        else:
            return [Dir.LEFT, Dir.STAY]

    def get_best_action(self, state):
        # funkcja zwracająca najlepszą akcję dla danego stanu
        possible_actions = self.get_possible_actions(state)
        return max(possible_actions, key=lambda action: self.get_q_value(state, action)) # nie działa bo nie jest dokończone get_q_value

    def get_action(self, state):
        # funkcja zwracająca akcję dla danego stanu
        # zgodnie z polityką epsilon-zachłanną
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(state))
        else:
            return self.get_best_action(state)

    def num_game(self):
        # funkcja zwracająca liczbę rozegranych gier
        return self.num_games

    def max_num_game(self):
        # funkcja zwracająca maksymalną liczbę gier do rozegrania
        return self.max_num_games

    def train(self):
        # funkcja trenująca agenta na kolejnej grze
        # TODO: trzeba dokończyć
        self.num_games += 1

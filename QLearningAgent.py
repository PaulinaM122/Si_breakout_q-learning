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
        self.reward_history = [0] * self.max_num_games
        self.success_history = []

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

    def num_game(self):
        # funkcja zwracająca liczbę rozegranych gier
        return self.num_games

    def max_num_game(self):
        # funkcja zwracająca maksymalną liczbę gier do rozegrania
        return self.max_num_games

    def increase_num_games(self):
        # funkcja inkrementująca liczbę rozgrywek wykonanych przez agenta
        self.num_games += 1

    def evaluate(self):
        # TODO: możemy tu dać np obliczanie średnich, żeby zobaczyć, jak zmienić parametry w treningu i na koniec do tworzenia staystyk/wykresów
        success_rate = sum(self.success_history) / self.num_games
        avg_reward = sum(self.reward_history) / self.num_games
        return success_rate, avg_reward

    def save_q_values(self):
        # funkcja zapisująca wartości wytrenowanych q_values do pliku q_values.txt
        # TODO: zapisać do pliku, żeby można było go było późńiej trenować od tych wartości
        pass

    def load_q_values(self):
        # funkcja wczytująca wartości wytrenowanych q_values z pliku q_values.txt
        # TODO: wczytanie z pliku q_values
        pass

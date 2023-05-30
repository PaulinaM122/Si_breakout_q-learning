import abc
from globals.constants import *
import numpy as np

class Agent(abc.ABC):
    def __init__(self):
        self.num_games = 0  # licznik gier
        self.max_num_games = MAX_NUM_GAMES  # maksymalna liczba gier do rozegrania
        self.reward_history = [0] * self.max_num_games
        self.success_history = []

    @abc.abstractmethod
    def get_action(self):
        pass

    def num_game(self):
        # funkcja zwracająca liczbę rozegranych gier
        return self.num_games

    def max_num_game(self):
        # funkcja zwracająca maksymalną liczbę gier do rozegrania
        return self.max_num_games

    def increase_num_games(self):
        # funkcja inkrementująca liczbę rozgrywek wykonanych przez agenta
        self.num_games += 1

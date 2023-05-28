import random
from agents.q_learning_agent import QLearningAgent
from globals.constants import *




class SarsaAgent(QLearningAgent):
    def __init__(self):
        self.prev_state = None
        self.prev_action = None
        super().__init__(q_values_file_name='q_values_sarsa.txt', screen_width=SCREEN_WIDTH_BIG)


    def update_q_value(self, state, action, next_state, next_action, reward, paddle):
        if self.prev_state is not None and self.prev_action is not None:
            current_q = self.get_q_value(self.prev_state, self.prev_action)
            next_q = self.get_q_value(next_state, next_action)
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_values[(self.prev_state, self.prev_action)] = new_q


    def get_best_action(self, state, paddle):
        possible_actions = self.get_possible_actions(paddle)
        max_value = max([self.get_q_value(state, action) for action in possible_actions])
        max_actions = [action for action in possible_actions if self.get_q_value(state, action) == max_value]
        if len(max_actions) > 1:
            return random.choice(max_actions)
        else:
            return max_actions[0]

    def get_action(self, state, paddle):
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(paddle))
        else:
            return self.get_best_action(state, paddle)

    def get_state(self, ball, paddle, bricks):
        return ball.get_relative_position(paddle), ball.get_direction()






from agents.q_learning_agent import QLearningAgent
from globals.constants import *


class SimpleAgent(QLearningAgent):
    def __init__(self):
        super().__init__(q_values_file_name='q_values_simple.txt', screen_width=SCREEN_WIDTH_BIG)

    def get_state(self, ball, paddle, bricks):
        return ball.get_relative_position(paddle), ball.get_direction()

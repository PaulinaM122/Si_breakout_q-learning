import json
import os
import re
from globals.direction import Direction as Dir
from agents.q_learning_agent import QLearningAgent


class ComplexAgent(QLearningAgent):
    def __init__(self):
        super().__init__(q_values_file_name='q_values_complex.txt')

    def get_state(self, ball, paddle, bricks):
        return ball.pos(), paddle.pos(), bricks.get_state()

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
                # Parsowanie klucza w formacie "(((10.00,-230.00), (0.00,-280.00), 36), <Direction.LEFT: 2>)"
                match = re.search(
                    r'\(\((-?\d+\.\d+),(-?\d+\.\d+)\), \((-?\d+\.\d+),(-?\d+\.\d+)\), (\d+)\), <Direction\.[A-Z_]+: (\d+)>',
                    key)
                ballx = eval(match.group(1))
                bally = eval(match.group(2))
                paddlex = eval(match.group(3))
                paddley = eval(match.group(4))
                bricks = eval(match.group(5))
                action = Dir(int(eval(match.group(6))))

                # Parsowanie wartości jako float
                q_value = float(value)
                ballPos = (ballx, bally)
                paddlePos = (paddlex, paddley)
                # Przypisanie wartości do słownika q_values
                self.q_values[(ballPos, paddlePos, bricks), action] = q_value

        return self.q_values
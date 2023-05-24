from agents.q_learning_agent import QLearningAgent


class ComplexAgent(QLearningAgent):
    def __init__(self):
        super().__init__(q_values_file_name='q_values_complex.txt')

    def get_state(self, ball, paddle, bricks):
        return ball.pos(), paddle.pos(), bricks.get_state()

from agent import Agent
from Direction import Direction as Dir


class FuzzyAgent(Agent):
    def __init__(self, width=590):
        # TODO: dobrać parametry tak, żeby nie wpadał w pętlę
        self.width = width
        self.paddle_left_mf = self.triangular(-width/2 - 10, -width/2, -width/4)
        self.paddle_center_mf = self.triangular(-width/4 - 50, 0, width/4 + 50)
        self.paddle_right_mf = self.triangular(width/4, width/2, width/2 + 10)

        self.ball_far_left_mf = self.triangular(-width/2 - 10, -width/2, -width/4 - 50)
        self.ball_left_mf = self.trapezoidal(-width / 3, -width / 4, -width / 6, -width / 12)
        self.ball_center_mf = self.triangular(-width/6 - 30, 0, width/6 + 30)
        self.ball_right_mf = self.trapezoidal(width / 12, width / 6, width / 4, width / 3)
        self.ball_far_right_mf = self.triangular(width/4 + 50, width/2, width/2 + 10)

        self.possible_actions = {
            Dir.LEFT.value: 0.0,
            Dir.STAY.value: 0.0,
            Dir.RIGHT.value: 0.0
        }

        self.paddle_pos = {
            'left': 0.0,
            'center': 0.0,
            'right': 0.0
        }

        self.ball_pos = {
            'far_left': 0.0,
            'left': 0.0,
            'center': 0.0,
            'right': 0.0,
            'far_right': 0.0
        }
        super().__init__()

    def get_action(self):
        return max(self.possible_actions, key=lambda k: self.possible_actions[k])

    def linear_increasing(self, x1, x2, x, low, high):
        return x * (low - high) / (x1 - x2) + (low - x1 * (low - high)/(x1 - x2))

    def linear_decreasing(self, x1, x2, x, low, high):
        return x * (high - low) / (x1 - x2) + (high - x1 * (high - low) / (x1 - x2))

    def triangular(self, x1, x2, x3, low=0.0, high=1.0) -> callable:
        assert x1 <= x2 <= x3

        def f(x):
            if x <= x1 or x >= x3:
                return low
            elif x1 < x < x2:
                return self.linear_increasing(x1, x2, x, low, high)
            else:
                return self.linear_decreasing(x2, x3, x, low, high)
        return f

    def trapezoidal(self, x1, x2, x3, x4, low=0.0, high=1.0) -> callable:
        assert x1 <= x2 <= x3 <= x4

        def f(x):
            if x <= x1 or x >= x4:
                return low
            elif x1 < x < x2:
                return self.linear_increasing(x1, x2, x, low, high)
            elif x2 <= x <= x3:
                return high
            else:
                return self.linear_decreasing(x3, x4, x, low, high)

        return f

    def calculate_membership(self, current_ball_pos, current_paddle_pos):
        self.paddle_pos['left'] = self.paddle_left_mf(current_paddle_pos)
        self.paddle_pos['center'] = self.paddle_center_mf(current_paddle_pos)
        self.paddle_pos['right'] = self.paddle_right_mf(current_paddle_pos)

        self.ball_pos['far_left'] = self.ball_far_left_mf(current_ball_pos)
        self.ball_pos['left'] = self.ball_left_mf(current_ball_pos)
        self.ball_pos['center'] = self.ball_center_mf(current_ball_pos)
        self.ball_pos['right'] = self.ball_right_mf(current_ball_pos)
        self.ball_pos['far_right'] = self.ball_far_right_mf(current_ball_pos)

    def inference(self):
        left = []
        stay = []
        right = []

        # action - stay
        # if paddle center and ball center
        stay.append(min(self.paddle_pos['center'], self.ball_pos['center']))
        # if paddle left and ball left
        stay.append(min(self.paddle_pos['left'], self.ball_pos['left']))
        # if paddle right and ball right
        stay.append(min(self.paddle_pos['right'], self.ball_pos['right']))

        # action - left
        # if paddle left and ball far left
        left.append(min(self.paddle_pos['left'], self.ball_pos['far_left']))
        # if paddle center and ball far left
        left.append(min(self.paddle_pos['center'], self.ball_pos['far_left']))
        # if paddle center and ball left
        left.append(min(self.paddle_pos['center'], self.ball_pos['left']))
        # if paddle right and ball far left
        left.append(min(self.paddle_pos['right'], self.ball_pos['far_left']))
        # if paddle right and ball left
        left.append(min(self.paddle_pos['right'], self.ball_pos['left']))
        # if paddle right and ball center
        left.append(min(self.paddle_pos['right'], self.ball_pos['center']))

        # action - right
        # if paddle right and ball far right
        right.append(min(self.paddle_pos['right'], self.ball_pos['far_right']))
        # if paddle center and ball far right
        right.append(min(self.paddle_pos['center'], self.ball_pos['far_right']))
        # if paddle center and ball right
        right.append(min(self.paddle_pos['center'], self.ball_pos['right']))
        # if paddle left and ball far right
        right.append(min(self.paddle_pos['left'], self.ball_pos['far_right']))
        # if paddle left and ball right
        right.append(min(self.paddle_pos['left'], self.ball_pos['right']))
        # if paddle left and ball center
        right.append(min(self.paddle_pos['left'], self.ball_pos['center']))

        self.possible_actions[Dir.LEFT.value] = max(left)
        self.possible_actions[Dir.STAY.value] = max(stay)
        self.possible_actions[Dir.RIGHT.value] = max(right)

    def plot_membership_functions(self):
        # TODO: wykresy f przynależności
        pass

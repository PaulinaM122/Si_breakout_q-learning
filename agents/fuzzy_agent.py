from agents.agent import *
from globals.direction import Direction as Dir
from globals.constants import *
import matplotlib.pyplot as plt


class FuzzyAgent(Agent):
    def __init__(self, screen_width=SCREEN_WIDTH_BIG):
        self.screen_width = screen_width
        self.paddle_far_left_mf = self.triangular(-screen_width / 2 - 10, -screen_width / 2, -screen_width / 4 - 120)
        self.paddle_left_mf = self.trapezoidal(-screen_width / 3 - 100, -screen_width / 4, -screen_width / 6, -screen_width / 12 + 100)
        self.paddle_center_mf = self.triangular(-screen_width / 4 + 135, 0, screen_width / 4 - 135)
        self.paddle_right_mf = self.trapezoidal(screen_width / 12 - 100, screen_width / 6, screen_width / 4, screen_width / 3 + 100)
        self.paddle_far_right_mf = self.triangular(screen_width / 4 + 120, screen_width / 2, screen_width / 2 + 10)

        self.ball_far_left_mf = self.triangular(-screen_width / 2 - 10, -screen_width / 2, -screen_width / 4 - 120)
        self.ball_left_mf = self.trapezoidal(-screen_width / 3 - 100, -screen_width / 4, -screen_width / 6, -screen_width / 12 + 120)
        self.ball_center_mf = self.triangular(-screen_width / 6 + 100, 0, screen_width / 6 - 100)
        self.ball_right_mf = self.trapezoidal(screen_width / 12 - 120, screen_width / 6, screen_width / 4, screen_width / 3 + 100)
        self.ball_far_right_mf = self.triangular(screen_width / 4 + 120, screen_width / 2, screen_width / 2 + 10)

        self.possible_actions = {
            Dir.LEFT.value: 0.0,
            Dir.STAY.value: 0.0,
            Dir.RIGHT.value: 0.0
        }

        self.paddle_pos = {
            'far_left': 0.0,
            'left': 0.0,
            'center': 0.0,
            'right': 0.0,
            'far_right': 0.0
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
        self.paddle_pos['far_left'] = self.paddle_far_left_mf(current_paddle_pos)
        self.paddle_pos['left'] = self.paddle_left_mf(current_paddle_pos)
        self.paddle_pos['center'] = self.paddle_center_mf(current_paddle_pos)
        self.paddle_pos['right'] = self.paddle_right_mf(current_paddle_pos)
        self.paddle_pos['far_right'] = self.paddle_far_right_mf(current_paddle_pos)

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
        # if paddle far left and ball far left
        stay.append(min(self.paddle_pos['far_left'], self.ball_pos['far_left'])) # 16
        # if paddle far right and ball far right
        stay.append(min(self.paddle_pos['far_right'], self.ball_pos['far_right'])) # 17

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
        # if paddle far right and ball far left
        left.append(min(self.paddle_pos['far_right'], self.ball_pos['far_left'])) # 18
        # if paddle far right and ball left
        left.append(min(self.paddle_pos['far_right'], self.ball_pos['left'])) # 19
        # if paddle far right and ball center
        left.append(min(self.paddle_pos['far_right'], self.ball_pos['center'])) # 20
        # if paddle far right and ball right
        left.append(min(self.paddle_pos['far_right'], self.ball_pos['right'])) # 21

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
        # if paddle far left and ball far right
        right.append(min(self.paddle_pos['far_left'], self.ball_pos['far_right'])) #22
        # if paddle far left and ball right
        right.append(min(self.paddle_pos['far_left'], self.ball_pos['right'])) #23
        # if paddle far left and ball center
        right.append(min(self.paddle_pos['far_left'], self.ball_pos['center'])) #24
        # if paddle far left and ball left
        right.append(min(self.paddle_pos['far_left'], self.ball_pos['left'])) #25

        self.possible_actions[Dir.LEFT.value] = max(left)
        self.possible_actions[Dir.STAY.value] = max(stay)
        self.possible_actions[Dir.RIGHT.value] = max(right)

    def plot_membership_functions(self):
        x = np.linspace(-self.screen_width/2, self.screen_width/2, 1200)
        y_paddle_far_left = [0] * int(len(x))
        y_paddle_left = [0] * int(len(x))
        y_paddle_center = [0] * int(len(x))
        y_paddle_right = [0] * int(len(x))
        y_paddle_far_right = [0] * int(len(x))
        y_ball_far_left = [0] * int(len(x))
        y_ball_left = [0] * int(len(x))
        y_ball_center = [0] * int(len(x))
        y_ball_right = [0] * int(len(x))
        y_ball_far_right = [0] * int(len(x))

        for i in range(int(len(x))):
            y_paddle_far_left[i] = self.paddle_far_left_mf(x[i])
            y_paddle_left[i] = self.paddle_left_mf(x[i])
            y_paddle_center[i] = self.paddle_center_mf(x[i])
            y_paddle_right[i] = self.paddle_right_mf(x[i])
            y_paddle_far_right[i] = self.paddle_far_right_mf(x[i])
            y_ball_far_left[i] = self.ball_far_left_mf(x[i])
            y_ball_left[i] = self.ball_left_mf(x[i])
            y_ball_center[i] = self.ball_center_mf(x[i])
            y_ball_right[i] = self.ball_right_mf(x[i])
            y_ball_far_right[i] = self.ball_far_right_mf(x[i])


        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x, y_paddle_far_left, 'b', linewidth=1.5)
        plt.plot(x, y_paddle_left)
        plt.plot(x, y_paddle_center)
        plt.plot(x, y_paddle_right)
        plt.plot(x, y_paddle_far_right)
        plt.title('Pozycja paletki')
        plt.xlabel('Położenie')
        plt.ylabel('Stopień przynależności')
        plt.legend(['dalekie lewo', 'lewo', 'środek', 'prawo', 'dalekie prawo'])

        plt.subplot(1, 2, 2)
        plt.plot(x, y_ball_far_left, 'b', linewidth=1.5)
        plt.plot(x, y_ball_left)
        plt.plot(x, y_ball_center)
        plt.plot(x, y_ball_right)
        plt.plot(x, y_ball_far_right)
        plt.title('Pozycja piłki')
        plt.xlabel('Położenie')
        plt.ylabel('Stopień przynależności')
        plt.legend(['dalekie lewo', 'lewo', 'środek', 'prawo', 'dalekie prawo'])

        plt.show()

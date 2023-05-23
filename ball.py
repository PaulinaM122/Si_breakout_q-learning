from turtle import Turtle
from Direction import Direction as Dir

MOVE_DIST = 10


class Ball(Turtle):
    def __init__(self):
        super().__init__()
        self.shape('circle')
        self.color('white')
        self.penup()
        self.x_move_dist = MOVE_DIST
        self.y_move_dist = MOVE_DIST
        self.reset()
        self.last_position = None
        self.current_position = self.position()

    def move(self):
        self.last_position = self.current_position
        new_y = self.ycor() + self.y_move_dist
        new_x = self.xcor() + self.x_move_dist
        self.goto(x=new_x, y=new_y)
        self.current_position = self.position()

    def bounce(self, x_bounce, y_bounce):
        if x_bounce:
            self.x_move_dist *= -1

        if y_bounce:
            self.y_move_dist *= -1

    def bounce_paddle(self, x_bounce, y_bounce):
        if x_bounce:
            self.x_move_dist *= -1
        self.y_move_dist = MOVE_DIST

    def bounce_upper_wall(self, x_bounce, y_bounce):
        if x_bounce:
            self.x_move_dist *= -1
        self.y_move_dist = -MOVE_DIST

    def reset(self):
        self.goto(x=0, y=-240)
        self.x_move_dist = MOVE_DIST
        self.y_move_dist = MOVE_DIST
        self.last_position = None
        self.current_position = self.position()

    def next_move(self):
        ycor = self.ycor() + self.y_move_dist
        xcor = self.xcor() + self.x_move_dist
        return xcor, ycor

    def get_relative_position(self, paddle):
        half_paddle = 110
        if self.xcor() > paddle.xcor() - half_paddle and self.xcor() < paddle.xcor() + half_paddle:
            relative_ball_pos = Dir.STAY
        elif self.xcor() < paddle.xcor():
            relative_ball_pos = Dir.LEFT
        else:
            relative_ball_pos = Dir.RIGHT
        return relative_ball_pos

    def get_direction(self):
        if self.x_move_dist < 0 and self.y_move_dist < 0:
            ball_dir = Dir.DOWN_LEFT
        elif self.x_move_dist < 0 and self.y_move_dist >= 0:
            ball_dir = Dir.UP_LEFT
        elif self.x_move_dist >= 0 and self.y_move_dist < 0:
            ball_dir = Dir.DOWN_RIGHT
        else:
            ball_dir = Dir.UP_RIGHT
        return ball_dir

    def get_distance(self):
        height = 600
        if self.ycor() < height / 3 - height/2:
            ball_distance = 0
        elif self.ycor() < height / 3 * 2 - height/2:
            ball_distance = 1
        else:
            ball_distance = 2
        return ball_distance

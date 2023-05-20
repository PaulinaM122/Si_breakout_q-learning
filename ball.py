from turtle import Turtle

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
        self.y_move_dist = MOVE_DIST

    def next_move(self):
        ycor = self.ycor() + self.y_move_dist
        xcor = self.xcor() + self.x_move_dist
        return xcor, ycor

import turtle as tr

from QLearningAgent import QLearningAgent
from paddle import Paddle
from ball import Ball
from scoreboard import Scoreboard
from ui import UI
from bricks import Bricks
from Direction import Direction as Dir
import time

screen = tr.Screen()
screen.setup(width=590, height=600)
screen.bgcolor('black')
screen.title('Breakout')
screen.tracer(0)

ui = UI()
ui.header()

score = Scoreboard(lives=3)
paddle = Paddle()
bricks = Bricks()

ball = Ball()

BALL_RADIOUS = 5
PADDLE_WIDTH = 220

game_paused = False
playing_game = True


def pause_game():
    global game_paused
    if game_paused:
        game_paused = False
    else:
        game_paused = True


screen.listen()
screen.onkey(key='Left', fun=paddle.move_left)
screen.onkey(key='Right', fun=paddle.move_right)
screen.onkey(key='space', fun=pause_game)


def check_collision_with_walls():
    global ball

    # detect collision with left and right walls:
    if ball.xcor() < -275 or ball.xcor() > 265:
        ball.bounce(x_bounce=True, y_bounce=False)
        return

    # detect collision with upper wall
    if ball.ycor() > 270:
        ball.bounce(x_bounce=False, y_bounce=True)
        return


def check_collision_with_bottom_wall():
    global ball, score, playing_game, ui

    # In this case, user failed to hit the ball
    # thus he loses. The game resets.
    if ball.ycor() <= -280:
        ball.reset()
        score.decrease_lives()
        if score.lives == 0:
            score.increase_game()
            score.reset()
            playing_game = False
            ui.game_over(win=False)
        ui.change_color()
        return True
    return False


def check_collision_with_paddle_left():
    paddle_left_edge = paddle.xcor() - PADDLE_WIDTH / 2
    paddle_top_edge = paddle.ycor() + BALL_RADIOUS
    paddle_bottom_edge = paddle.ycor() - BALL_RADIOUS

    ball_x, ball_y = ball.next_move()
    ball_right_edge = ball_x + BALL_RADIOUS
    ball_top_edge = ball_y + BALL_RADIOUS
    ball_bottom_edge = ball_y - BALL_RADIOUS

    if (
        ball_right_edge >= paddle_left_edge and
        ball_top_edge <= paddle_top_edge and
        ball_bottom_edge >= paddle_bottom_edge
    ):
        return True

    return False


def check_collision_with_paddle_right():
    paddle_right_edge = paddle.xcor() + PADDLE_WIDTH / 2
    paddle_top_edge = paddle.ycor() + BALL_RADIOUS
    paddle_bottom_edge = paddle.ycor() - BALL_RADIOUS

    ball_x, ball_y = ball.next_move()
    ball_left_edge = ball_x - BALL_RADIOUS
    ball_top_edge = ball_y + BALL_RADIOUS
    ball_bottom_edge = ball_y - BALL_RADIOUS

    if (
        ball_left_edge <= paddle_right_edge and
        ball_top_edge <= paddle_top_edge and
        ball_bottom_edge >= paddle_bottom_edge
    ):
        return True

    return False


def check_collision_with_paddle():
    global ball, paddle
    # record x-axis coordinates of ball and paddle
    paddle_x = paddle.xcor()
    ball_x = ball.xcor()

    # check if ball's distance(from its middle)
    # from paddle(from its middle) is less than
    # width of paddle and ball is below a certain
    # coordinate to detect their collision
    if ball.distance(paddle) < (110 + BALL_RADIOUS) and ball.ycor() < -250:

        # If Paddle is on Right of Screen
        if paddle_x > 0:
            if ball_x > paddle_x:
                # If ball hits paddles left side it
                # should go back to left
                ball.bounce(x_bounce=True, y_bounce=True)
                return
            else:
                ball.bounce(x_bounce=False, y_bounce=True)
                return

        # If Paddle is left of Screen
        elif paddle_x < 0:
            if ball_x < paddle_x:
                # If ball hits paddles left side it
                # should go back to left
                ball.bounce(x_bounce=True, y_bounce=True)
                return
            else:
                ball.bounce(x_bounce=False, y_bounce=True)
                return

        # Else Paddle is in the Middle horizontally
        else:
            if ball_x > paddle_x:
                ball.bounce(x_bounce=True, y_bounce=True)
                return
            elif ball_x < paddle_x:
                ball.bounce(x_bounce=True, y_bounce=True)
                return
            else:
                ball.bounce(x_bounce=False, y_bounce=True)
                return


def check_collision_with_bricks():
    # returns True/False
    # TODO: tu jest zdecydowanie coś źle z tym sprawdzaniem, bo czasem zbija dużo po skosie, ale możemy tak zostawić xd
    global ball, score, bricks
    collided = False

    for brick in bricks.bricks:
        if ball.distance(brick) < 40:
            collided = True
            score.increase_score()
            brick.clear()
            brick.goto(3000, 3000)
            bricks.bricks.remove(brick)

            # detect collision from left
            if ball.xcor() < brick.left_wall:
                ball.bounce(x_bounce=True, y_bounce=False)

            # detect collision from right
            elif ball.xcor() > brick.right_wall:
                ball.bounce(x_bounce=True, y_bounce=False)

            # detect collision from bottom
            elif ball.ycor() < brick.bottom_wall:
                ball.bounce(x_bounce=False, y_bounce=True)

            # detect collision from top
            elif ball.ycor() > brick.upper_wall:
                ball.bounce(x_bounce=False, y_bounce=True)

            break

    return collided


# create Q-learning agent
agent = QLearningAgent()

while True:
    # reset the game
    score.lives = 3
    bricks.reset()
    ball.reset()
    paddle.goto(x=0, y=-280)
    game_paused = False
    playing_game = True

    # start a new game
    while playing_game:
        if not game_paused:

            reward = 0
            state = agent.get_state(ball, paddle, bricks)
            action = agent.get_action(state)

            # perform action
            if action == Dir.LEFT:
                if not check_collision_with_paddle_left():
                    paddle.move_left()
            elif action == Dir.RIGHT:
                if not check_collision_with_paddle_right():
                    paddle.move_right()

            # UPDATE SCREEN WITH ALL THE MOTION THAT HAS HAPPENED
            screen.update()
            time.sleep(0.01)
            ball.move()

            # DETECTING COLLISION WITH WALLS
            check_collision_with_walls()
            if check_collision_with_bottom_wall():
                reward += -100

            # DETECTING COLLISION WITH THE PADDLE
            if check_collision_with_paddle():
                reward += 30

            # DETECTING COLLISION WITH A BRICK
            if check_collision_with_bricks():
                # get reward for breaking the brick
                reward += 1

            # DETECTING USER'S VICTORY
            if len(bricks.bricks) == 0:
                ui.game_over(win=True)
                playing_game = False
                reward += 100

            # update Q-values
            next_state = agent.get_state(ball, paddle, bricks)
            agent.update_q_value(state, action, next_state, reward)

            # Ball reset if error
            if(
                    ball.xcor() > 350 or
                    ball.xcor() < -350 or
                    ball.ycor() < -350 or
                    ball.ycor() > 350
            ):
                ball.reset()

    # check if the agent won the game
    if len(bricks.bricks) == 0:
        ui.game_over(win=True)
    else:
        ui.game_over(win=False)
        # agent lost the game, start over
        continue

    # check if the maximum number of games has been reached
    if agent.num_games >= agent.max_num_games:
        break

    # train the agent for the next game
    agent.train()

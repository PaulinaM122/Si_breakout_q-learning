import turtle as tr

import matplotlib.pyplot as plt
import numpy as np

from items.paddle import Paddle
from items.ball import Ball
from view.scoreboard import Scoreboard
from view.ui import UI
from items.bricks import Bricks
from agents.simple_agent import SimpleAgent
from globals.direction import Direction as Dir
from globals.constants import *
import time

width = SCREEN_WIDTH_BIG
height = SCREEN_HEIGHT

max_bricks = MAX_BRICKS_BIG
if width == SCREEN_WIDTH_SMALL:
    max_bricks = MAX_BRICKS_SMALL

screen = tr.Screen()
screen.setup(width=width, height=height)
screen.bgcolor('black')
screen.title('Breakout')
screen.tracer(0)

ui = UI()
ui.header()

score = Scoreboard(lives=3)
paddle = Paddle()
bricks = Bricks(screen_width=width)

ball = Ball()

playing_game = True
training_agent = True

# create Q-learning agent
agent = SimpleAgent()
agent.load_q_values()


def check_collision_with_walls():
    global width, height, ball

    # detect collision with left and right walls:
    edge = width/2 - BALL_DIAMETER

    if ball.xcor() < -edge and ball.x_move_dist < 0 or ball.xcor() > edge and ball.x_move_dist > 0:
        ball.bounce(x_bounce=True, y_bounce=False)
        return

    # detect collision with upper wall
    ceiling = height/2 - BALL_DIAMETER
    if ball.ycor() > ceiling and ball.y_move_dist > 0:
        ball.bounce_upper_wall(x_bounce=False, y_bounce=True)
        return


def check_collision_with_bottom_wall():
    global ball, score, playing_game, ui, height

    # In this case, user failed to hit the ball
    # thus he loses. The game resets.
    if ball.ycor() < -(height/2 - BALL_DIAMETER) and ball.y_move_dist < 0:
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


def check_collision_with_paddle():
    global ball, paddle
    # record x-axis coordinates of ball and paddle
    paddle_x = paddle.xcor()
    ball_x = ball.xcor()

    # check if ball's distance(from its middle)
    # from paddle(from its middle) is less than
    # width of paddle and ball is below a certain
    # coordinate to detect their collision
    if ball.distance(paddle) <= HALF_PADDLE \
            and ball.ycor() <= PADDLE_Y_POS + PADDLE_HEIGHT \
            and ball.y_move_dist < 0:

        # If Paddle is on Right of Screen
        if paddle_x > 0:
            if ball_x > paddle_x:
                # If ball hits paddles left side it
                # should go back to left
                ball.bounce(x_bounce=True, y_bounce=True)
                return True
            else:
                ball.bounce(x_bounce=False, y_bounce=True)
                return True

        # If Paddle is left of Screen
        elif paddle_x < 0:
            if ball_x < paddle_x:
                # If ball hits paddles left side it
                # should go back to left
                ball.bounce(x_bounce=True, y_bounce=True)
                return True
            else:
                ball.bounce(x_bounce=False, y_bounce=True)
                return True

        # Else Paddle is in the Middle horizontally
        else:
            if ball_x > paddle_x:
                ball.bounce(x_bounce=True, y_bounce=True)
                return True
            elif ball_x < paddle_x:
                ball.bounce(x_bounce=True, y_bounce=True)
                return True
            else:
                ball.bounce(x_bounce=False, y_bounce=True)
                return True
    return False


def check_collision_with_bricks():
    global ball, score, bricks

    for brick in bricks.bricks:
        if ball.distance(brick) < 40:
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

            return


def reset_the_game():
    global score, bricks, ball, paddle, playing_game

    score.lives = 3
    bricks.reset()
    ball.reset()
    paddle.goto(x=0, y=PADDLE_Y_POS)
    playing_game = True


def leave_game():
    global playing_game, training_agent
    playing_game = False
    training_agent = False


screen.listen()
screen.onkey(key='Left', fun=paddle.move_left)
screen.onkey(key='Right', fun=paddle.move_right)
screen.onkey(key='Escape', fun=leave_game)


while training_agent:
    reset_the_game()

    # start a new game
    while playing_game:
        reward = 0
        ball.move()
        state = agent.get_state(ball, paddle, bricks)
        action = agent.get_action(state, paddle)

        # perform action
        if action == Dir.LEFT:
            paddle.move_left()
        elif action == Dir.RIGHT:
            paddle.move_right()
        else:
            reward += 1

        # UPDATE SCREEN WITH ALL THE MOTION THAT HAS HAPPENED
        screen.update()
        #time.sleep(0.01)

        # DETECTING COLLISION WITH WALLS
        check_collision_with_walls()
        if check_collision_with_bottom_wall():
            reward += -1 * abs(ball.xcor() - paddle.xcor())
            agent.success_history.append(0)

        # DETECTING COLLISION WITH THE PADDLE
        if check_collision_with_paddle():
            reward += 100

        # DETECTING COLLISION WITH A BRICK
        check_collision_with_bricks()

        # DETECTING USER'S VICTORY
        if len(bricks.bricks) == 0:
            ui.game_over(win=True)
            playing_game = False
            agent.success_history.append(1)

        agent.reward_history[agent.num_games] += reward

        # update Q-values
        next_state = agent.get_state(ball, paddle, bricks)
        agent.update_q_value(state, action, next_state, reward, paddle)

    # check if the agent won the game
    if len(bricks.bricks) == 0:
        ui.game_over(win=True)
    else:
        # agent lost the game, start over
        ui.game_over(win=False)

    ui.saving_q_values()
    screen.update()
    agent.save_q_values()

    agent.bricks_hit.append(max_bricks - len(bricks.bricks))

    agent.increase_num_games()
    print(agent.num_games)
    # check if the maximum number of games has been reached
    if agent.num_games >= agent.max_num_games:
        break

average = "{:.2f}".format(np.average(agent.bricks_hit))

plt.plot(agent.bricks_hit)
plt.title("Simple Q-learning - liczba zbitych klocków na koniec gry\nśrednia: " + str(average))
plt.xlabel("numer gry")
plt.ylabel("liczba zbitych klocków")
plt.savefig("charts/simple_agent.png")

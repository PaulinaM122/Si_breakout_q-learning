import turtle as tr

from QLearningAgent import QLearningAgent
from paddle import Paddle
from ball import Ball
from scoreboard import Scoreboard
from ui import UI
from bricks import Bricks
import time

screen = tr.Screen()
screen.setup(width=1200, height=600)
screen.bgcolor('black')
screen.title('Breakout')
screen.tracer(0)

ui = UI()
ui.header()

score = Scoreboard(lives=3)
paddle = Paddle()
bricks = Bricks()

ball = Ball()

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
    global ball, score, playing_game, ui

    # detect collision with left and right walls:
    if ball.xcor() < -580 or ball.xcor() > 570:
        ball.bounce(x_bounce=True, y_bounce=False)
        return

    # detect collision with upper wall
    if ball.ycor() > 270:
        ball.bounce(x_bounce=False, y_bounce=True)
        return

    # detect collision with bottom wall
    # In this case, user failed to hit the ball
    # thus he loses. The game resets.
    if ball.ycor() < -280:
        ball.reset()
        score.decrease_lives()
        if score.lives == 0:
            score.reset()
            playing_game = False
            ui.game_over(win=False)
            return
        ui.change_color()
        return


def check_collision_with_paddle():
    global ball, paddle
    # record x-axis coordinates of ball and paddle
    paddle_x = paddle.xcor()
    ball_x = ball.xcor()

    # check if ball's distance(from its middle)
    # from paddle(from its middle) is less than
    # width of paddle and ball is below a certain
    # coordinate to detect their collision
    if ball.distance(paddle) < 110 and ball.ycor() < -250:

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
    global ball, score, bricks

    for brick in bricks.bricks:
        if ball.distance(brick) < 40:
            score.increase_score()
            brick.quantity -= 1
            if brick.quantity == 0:
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

# create Q-learning agent
agent = QLearningAgent()

while True:  # brakuje nagrody ujemnej, trzeba zaimplementować dla lepszych rezultatów uczenia
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

            # UPDATE SCREEN WITH ALL THE MOTION THAT HAS HAPPENED
            screen.update()
            time.sleep(0.01)
            ball.move()

            # DETECTING COLLISION WITH WALLS
            check_collision_with_walls()

            # DETECTING COLLISION WITH THE PADDLE
            check_collision_with_paddle()

            # DETECTING COLLISION WITH A BRICK
            if check_collision_with_bricks():
                # get reward for breaking the brick
                reward = 1
                # update Q-values
                state = agent.get_state(ball, paddle, bricks)
                action = agent.get_action(state)
                next_state = agent.get_state(ball, paddle, bricks)
                agent.update_q_value(state, action, next_state, reward)

            # DETECTING USER'S VICTORY
            if len(bricks.bricks) == 0:
                ui.game_over(win=True)
                # get reward for winning the game
                reward = 1
                # update Q-values
                state = agent.get_state(ball, paddle, bricks)
                action = agent.get_action(state)
                agent.update_q_value(state, action, state, reward)
                playing_game = False

            # get current state and action
            ball_state = ball.pos()
            paddle_state = paddle.pos()
            bricks_state = bricks.get_state()
            state = (ball_state, paddle_state, bricks_state)
            action = agent.get_action(state)

            # update Q-values
            if state is not None and action is not None:
                agent.update_q_value(state, action, state, 0)

            # perform action
            if action == 'LEFT':
                paddle.move_left()
            elif action == 'RIGHT':
                paddle.move_right()

            # DETECTING COLLISION WITH A BRICK
            if check_collision_with_bricks():
                # get reward for breaking the brick
                reward = 1
                # update Q-values
                next_ball_state = ball.pos()
                next_paddle_state = paddle.pos()
                next_bricks_state = bricks.get_state()
                next_state = (next_ball_state, next_paddle_state, next_bricks_state)
                agent.update_q_value(state, action, next_state, reward)


        else:
            ui.paused_status()

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




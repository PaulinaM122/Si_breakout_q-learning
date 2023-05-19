import turtle as tr

from QLearningAgent import QLearningAgent
from paddle import Paddle
from ball import Ball
from scoreboard import Scoreboard
from ui import UI
from bricks import Bricks
from Direction import Direction as Dir
import time
import utilities

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

playing_game = True
training_agent = True


def leave_game():
    global playing_game, training_agent
    playing_game = False
    training_agent = False


# create Q-learning agent
agent = QLearningAgent()

screen.listen()
screen.onkey(key='Left', fun=paddle.move_left)
screen.onkey(key='Right', fun=paddle.move_right)
screen.onkey(key='Escape', fun=leave_game)
screen.onkey(key='l', fun=agent.load_q_values)


def check_collision_with_walls():
    global ball

    # detect collision with left and right walls:
    if ball.xcor() < -580 or ball.xcor() > 570:
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
    if ball.ycor() < -280:
        ball.reset()
        score.decrease_lives()
        if score.lives == 0:
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


while training_agent:

    # reset the game
    score.lives = 3
    bricks.reset()
    ball.reset()
    paddle.goto(x=0, y=-280)
    playing_game = True

    # start a new game
    while playing_game:
        reward = 0
        state = agent.get_state(ball, paddle, bricks)
        action = agent.get_action(state)

        # perform action
        if action == Dir.LEFT:
            paddle.move_left()
        elif action == Dir.RIGHT:
            paddle.move_right()

        # UPDATE SCREEN WITH ALL THE MOTION THAT HAS HAPPENED
        screen.update()
        time.sleep(0.01)
        ball.move()

        # DETECTING COLLISION WITH WALLS
        check_collision_with_walls()
        if check_collision_with_bottom_wall():
            reward += -100
            agent.success_history.append(0)

        # DETECTING COLLISION WITH THE PADDLE
        check_collision_with_paddle()

        # DETECTING COLLISION WITH A BRICK
        if check_collision_with_bricks():
            # get reward for breaking the brick
            reward += 1

        # DETECTING USER'S VICTORY
        if len(bricks.bricks) == 0:
            ui.game_over(win=True)
            playing_game = False
            reward += 100
            agent.success_history.append(1)

        agent.reward_history[agent.num_games] += reward

        # update Q-values
        next_state = agent.get_state(ball, paddle, bricks)
        agent.update_q_value(state, action, next_state, reward)

    # check if the agent won the game
    if len(bricks.bricks) == 0:
        ui.game_over(win=True)
    else:
        ui.game_over(win=False)
        # agent lost the game, start over

    agent.increase_num_games()
    # check if the maximum number of games has been reached
    if agent.num_games >= agent.max_num_games:
        break


print("Agent finished training")
print("Saving q values to file q_values.txt...")
ui.saving_q_values()
screen.update()
agent.save_q_values()
print("Done")

from turtle import Turtle

try:
    score = int(open('./database_files/highest_score.txt', 'r').read())
except FileNotFoundError:
    score = open('./database_files/highest_score.txt', 'w').write(str(0))
except ValueError:
    score = 0

try:
    with open('./database_files/game_number.txt', 'r') as file:
        game_number = int(file.read())
except FileNotFoundError:
    with open('./database_files/game_number.txt', 'w') as file:
        file.write(str(0))
    game_number = 0
except ValueError:
    game_number = 0

FONT = ('arial', 16, 'normal')


class Scoreboard(Turtle):
    def __init__(self, lives):
        super().__init__()
        self.color('white')
        self.penup()
        self.hideturtle()
        self.highScore = score
        self.goto(x=-280, y=260)
        self.lives = lives
        self.score = 0
        self.game_number = game_number
        self.update_score()

    def update_score(self):
        self.clear()
        self.write(f"Score: {self.score} | Highest Score: \
        {self.highScore} | Lives: {self.lives}  | Game: {self.game_number}", align='left', font=FONT)

    def increase_score(self):
        self.score += 1
        if self.score > self.highScore:
            self.highScore += 1
        self.update_score()

    def decrease_lives(self):
        self.lives -= 1
        self.update_score()

    def increase_game(self):
        self.game_number += 1
        self.update_score()

    def reset(self):
        self.clear()
        self.score = 0
        self.update_score()
        open('./database_files/highest_score.txt', 'w').write(str(self.highScore))
        open('./database_files/game_number.txt', 'w').write(str(self.game_number))

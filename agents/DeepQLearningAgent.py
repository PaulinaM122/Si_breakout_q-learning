import random
import numpy as np
import matplotlib.pyplot as plt  # Importowanie biblioteki matplotlib
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from agents.q_learning_agent import QLearningAgent
from globals.direction import Direction as Dir
from globals.constants import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


ACTION_SIZE = 3  # Left, Right and Stay
STATE_SIZE = 2 #  ball direction and relative position


class DeepQLearningAgent(QLearningAgent):
    def __init__(self):
        super().__init__(q_values_file_name='q_values_deep.txt', screen_width=SCREEN_WIDTH_SMALL)
        self.model = self.build_model()
        self.replay_memory = []  # pamięć powtórek, przechowuje state,q_value
        self.q_values_history = {}

    def get_state(self, ball, paddle, bricks):
        relative_position = [ball.get_relative_position(paddle).value]
        direction = ball.get_direction().value
        state_array = np.array(relative_position + [direction])
        return state_array

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(STATE_SIZE,), activation='relu'))  # pierwsza warstwa modelu
        model.add(Dense(64, activation='relu'))  # druga warstwa modelu
        model.add(Dense(ACTION_SIZE, activation='linear'))  # trzecia warstwa modelu
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))  # Kompiluj model z odpowiednią funkcją straty i optymalizatorem
        return model

    def get_q_value(self, state, action):
        state_array = np.array(state)  # wejscia sieci neuronowej
        q_values = self.model.predict(state_array.reshape(1, -1))  # predict oblicza wartość q-funkcji, wykonując propagację w przód przez sieć neuronową
        # propagacja w przód to proces przekazywania danych przez sieć od warstwy wejściowej do warstwy wyjściowej, gdzie każda warstwa wykonuje obliczenia na podstawie wag i funkcji aktywacji
        if state[0] == Dir.STAY.value:
            this_state = "STAY"
        elif state[0] == Dir.LEFT.value:
            this_state = "LEFT"
        elif state[0] == Dir.RIGHT.value:
            this_state = "RIGHT"
        if state[1] == Dir.UP_LEFT.value:
            this_state2 = "UP_LEFT"
        elif state[1] == Dir.UP_RIGHT.value:
            this_state2 = "UP_RIGHT"
        elif state[1] == Dir.DOWN_LEFT.value:
            this_state2 = "DOWN_LEFT"
        elif state[1] == Dir.DOWN_RIGHT.value:
            this_state2 = "DOWN_RIGHT"
        if action + 1 == Dir.STAY.value:
            this_action = "STAY"
        elif action + 1 == Dir.LEFT.value:
            this_action = "LEFT"
        elif action + 1 == Dir.RIGHT.value:
            this_action = "RIGHT"
        print(f" position: {this_state}, direction: {this_state2} for action {this_action} - q_value: {q_values[0][action]}")
        return q_values[0][action]  # wartość dla konkretnej akcji

    def update_q_value(self, state, action, next_state, reward, paddle):
        state_array = np.array(state)
        next_state_array = np.array(next_state)
        q_values = np.copy(self.model.predict(state_array.reshape(1, -1))[0])  # propagacja w przód
        next_q_values = self.model.predict(next_state_array.reshape(1, -1))[0]  # propagacja w przód
        q_values[action.value - 1] = reward + self.gamma * np.max(next_q_values)

        # dodaj do pamięci powtórek
        self.replay_memory.append((state_array.copy(), q_values.copy()))

        # Inicjalizacja q_values_history dla kombinacji (position, direction, action), jeśli jeszcze nie istnieje
        if not hasattr(self, 'q_values_history'):
            self.q_values_history = {}

        # Aktualizacja q_values_history dla kombinacji (position, direction, action)
        state_key = (state[0], state[1], action.value)
        if state_key not in self.q_values_history:
            self.q_values_history[state_key] = []
        self.q_values_history[state_key].append(q_values[action.value - 1])

        # trening sieci na losowej próbce z pamięci powtórek
        batch_size = 1
        if len(self.replay_memory) >= batch_size:
            batch = random.sample(self.replay_memory, batch_size)
            states = np.array([sample[0] for sample in batch])
            target_q_values = np.array([sample[1] for sample in batch])
            self.model.fit(states, target_q_values, epochs=10, verbose=0) # fit dokonuje aktualizacji wag sieci, aby minimalizować różnicę między przewidywanymi a oczekiwanymi wartościami Q-funkcji



    def get_best_action(self, state, paddle):
        possible_actions = self.get_possible_actions(paddle)
        possible_q_values = [self.get_q_value(state, action.value - 1) for action in possible_actions]

        max_q_value = np.max(possible_q_values)
        best_action_indices = np.where(np.isclose(possible_q_values, max_q_value))[0]

        if len(best_action_indices) > 1:
            return Dir.STAY
        else:
            best_action_index = random.choice(best_action_indices)

        return possible_actions[best_action_index]

    def generate_q_value_plots(self):
        positions = [Dir.LEFT, Dir.RIGHT, Dir.STAY]
        directions = [Dir.UP_LEFT, Dir.UP_RIGHT, Dir.DOWN_LEFT, Dir.DOWN_RIGHT]
        actions = [Dir.LEFT, Dir.RIGHT, Dir.STAY]

        for direction in directions:
            fig, ax = plt.subplots()
            ax.set_title(f'Direction: {direction.name}')

            for position in positions:
                for action in actions:
                    state_key = (position.value, direction.value, action.value)
                    if state_key in self.q_values_history:
                        q_values_history = self.q_values_history[state_key]
                        episodes = np.arange(1, len(q_values_history) + 1)

                        label = f'Position: {position.name}, Action: {action.name}'
                        ax.plot(episodes, q_values_history, label=label)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Q-value')
            ax.legend()

            plt.show()








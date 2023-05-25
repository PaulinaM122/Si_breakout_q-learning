import random
import numpy as np
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

    def get_state(self, ball, paddle, bricks):
        relative_position = [ball.get_relative_position(paddle).value]
        direction = ball.get_direction().value
        state_array = np.array(relative_position + [direction])
        return state_array

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(STATE_SIZE,), activation='relu'))  # pierwsza warstwa modelu
        model.add(Dense(64, activation='relu'))  # druga warstwa modelu
        model.add(Dense(ACTION_SIZE, activation='linear'))  # trzecia warstwa modelu
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))  # Kompiluj model z odpowiednią funkcją straty i optymalizatorem
        return model

    def get_q_value(self, state, action):
        state_array = np.array(state)  # wejscia sieci neuronowej
        q_values = self.model.predict(state_array.reshape(1, -1))  # predict oblicza wartość q-funkcji, wykonując propagację w przód przez sieć neuronową
        # propagacja w przód to proces przekazywania danych przez sieć od warstwy wejściowej do warstwy wyjściowej, gdzie każda warstwa wykonuje obliczenia na podstawie wag i funkcji aktywacji
        return q_values[0][action]  # wartość dla konkretnej akcji

    def update_q_value(self, state, action, next_state, reward, paddle):
        state_array = np.array(state)
        next_state_array = np.array(next_state)
        q_values = np.copy(self.model.predict(state_array.reshape(1, -1))[0])  # propagacja w przód
        next_q_values = self.model.predict(next_state_array.reshape(1, -1))[0]  # propagacja w przód
        q_values[action.value - 1] = reward + self.gamma * np.max(next_q_values)

        # dodaj do pamięci powtórek
        self.replay_memory.append((state_array, q_values))

        # trening sieci na losowej próbce z pamięci powtórek
        batch_size = 32
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



'''
- creating en environment that can be used for reinforcement leanring
- can use other models from
- add early stopping in validation testing


'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimisers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import datetime

class env:
    def __init__(self, data):
        #data is input training data?
        self.data = data
        self.reset()

    def reset(self):
        #initializes trader


    def step(self, action, val):
        # returns new position, reward,
        # whether or not state is terminal, and debug info (print statements)
        next_val = val*data[state+1] # the val multiplied by the percent change of the next data point
        reward = 0 # reward is the differential sharpe ratio
        _state = val * next_val

        if action == 1:
            # buy, sharpe ratio
        elif action == 2:
            # sell
        else:
            # neutral
        return reward, _state


class agent:
    def __init__(self, gamma=0.99, epsilon=0.3, epsilon_decay
                 input_shape,):
        # self.everything above

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.input_shape = input_shape


    def DQN(self):
        model = Sequential([
            Dense(128, input_shape=self.input_shape, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(3, activati='softmax')
        ])
        opt = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(optimizers=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
        )
        tensorboard = TensorBoard(log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        filepath = #filepath of weights... include epoch number and validation accuracy
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True)
        history = model.fit(x_train, y_train, batch_size=64, epochs=100,
                                validation_data=(validation_x, validation_y),
                                callbacks=(tensorboard, checkpoint))



        #use tensorboard and matplotlib to analyze data?
        #want to see validation loss/accuracy and model loss/accuracy



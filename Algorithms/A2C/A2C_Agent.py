import os
import numpy as np
import random
import time
from copy import copy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras import backend as K
import argparse
import tensorflow as tf
from operator import itemgetter

import sys

sys.path.extend(['../../../gym-guidance-collision-avoidance-single'])
sys.path.extend(['../../Simulators'])
from Simulators.SingleAircraftDiscrete3HEREnv import SingleAircraftDiscrete3HEREnv


################################
##                            ##
##      Marc Brittain         ##
##  marcbrittain.github.io    ##
##                            ##
################################


# initalize the A2C agent
class A2C_Agent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # initial epsilon value
        self.epsilon_vals = np.append(np.linspace(self.epsilon, 0.1, 1000), np.linspace(0.1, 0.01, 4000))

        # if mac
        # self.speeds = np.array([101,259])
        self.total_reward = 0
        self.batch_size = 32

        self.lr = 0.0005

        self.value_size = 1

        # saving our values from the flag episode
        self.model_check = []

        self.model = self._build_A2C()

        self.UPDATE_FREQ = 7000  # how often to update the target network

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        return np.array([np.npv((1 / self.gamma - 1), r[i:]) for i in range(len(r))])

        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r

        assert q[0] == discounted_r[0]
        return discounted_r

    def _build_A2C(self):

        I = tf.keras.layers.Input(shape=(self.state_size,), name='states')

        # # dgoal, dintersection, route, speed, acceleration, radius
        # own_state = tf.keras.layers.Lambda(lambda x: x[:, :4], output_shape=(4,))(I)
        #
        # # dgoal, dintersection, route, speed, acceleration, radius, downship,LOS
        # # other_state = tf.keras.layers.Lambda(lambda x: x[:, 4:], output_shape=(self.state_size-4,))(I)
        #
        # # encoding other_state into 32 values
        # H1_int = tf.keras.layers.Dense(32, activation='relu')(other_state)
        #
        # # now combine them
        # combined = tf.keras.layers.concatenate([own_state, H1_int], axis=-1)

        H2 = tf.keras.layers.Dense(256, activation='relu')(I)
        H3 = tf.keras.layers.Dense(256, activation='relu')(H2)

        output = tf.keras.layers.Dense(self.action_size + 1, activation=None)(H3)

        # Split the output layer into policy and value
        policy = tf.keras.layers.Lambda(lambda x: x[:, :self.action_size], output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:, self.action_size:], output_shape=(self.value_size,))(output)

        # now I need to apply activation

        policy_out = tf.keras.layers.Activation('softmax', name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear', name='value_out')(value)

        # Using Adam optimizer, RMSProp's successor.
        opt = tf.keras.optimizers.Adam(lr=self.lr)

        model = tf.keras.models.Model(inputs=I, outputs=[policy_out, value_out])

        # The model is trained on 2 different loss functions
        model.compile(optimizer=opt, loss={'policy_out': 'categorical_crossentropy', 'value_out': 'mse'})

        return model

    def train(self, paths):

        # convert into single loop
        states_obs = np.concatenate([_path["observation"] for _path in paths])
        actions_obs = np.concatenate([_path["action"] for _path in paths])
        train_status = np.concatenate([[_path['info']] for _path in paths])  # [n_success, n_out_map, n_conflict]

        # Compute Q values (reward to go)
        q_n = np.concatenate(
            [[np.npv((1 / self.gamma - 1), path["reward"][i:]) for i in range(len(path["reward"]))] for path in
             paths])
        q_n = q_n.reshape(-1, 1)
        # normalize baseline value to be the training target of the value network
        # q_n_mean = q_n.mean()
        # q_n_std = q_n.std()
        # q_n = (q_n - q_n_mean) / q_n_std
        # print(q_n.shape)
        _, value_output = self.model.predict(states_obs)
        adv_n_baseline = q_n - value_output

        # print(adv_n.shape)
        # adv_n = adv_n * q_n_std + q_n_mean
        adv_n = adv_n_baseline
        # print(adv_n.shape)
        # normalize advantage A
        # adv_n = (adv_n - adv_n.mean()) / adv_n.std()
        # print(adv_n.shape)
        # backprop the policy network

        advantages = np.zeros((states_obs.shape[0], self.action_size))
        for i in range(states_obs.shape[0]):
            advantages[i][actions_obs[i]] = adv_n[i]

        self.model.fit({'states': states_obs},
                       {'policy_out': advantages, 'value_out': q_n},
                       epochs=1, batch_size=64, verbose=0)

    def load(self, name):
        try:
            self.model.load_weights(name)
            print('model loaded successfully...')
        except OSError:
            print('did not find file %s' % name)
        except ValueError:
            print('load wrong model')

    def save(self, name):
        self.model.save_weights(name)

    # action implementation for the agent
    def act(self, state):
        state = state.reshape([1, self.state_size])
        policy, value = self.model.predict(state)
        a = np.random.choice(self.action_size, 1, p=policy.flatten())[0]

        return a

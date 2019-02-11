#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import gym
import random
import os
from collections import deque
from matplotlib import pyplot as plt

from gym_guidance_collision_avoidance_single.envs import SingleAircraftDiscreteHEREnv

np.set_printoptions(suppress=True)
seed = 2
np.random.seed(2)
random.seed(2)


# feed forward neural network
class Model:
    def __init__(self, input, name):
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(shape=[None, input + 2], dtype=tf.float32)
            init = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                  mode='FAN_AVG',
                                                                  uniform=False)
            self.hidden1 = fully_connected_layer(self.inputs, 256,
                                                 activation=tf.nn.relu, init=init, scope='fc1')
            self.hidden2 = fully_connected_layer(self.hidden1, 256,
                                                 activation=tf.nn.relu, init=init, scope='fc2')
            self.Q_ = fully_connected_layer(self.hidden2, 3, activation=None, scope='Q', bias=False)
            self.predict = tf.argmax(self.Q_, axis=-1)
            self.action = tf.placeholder(shape=None, dtype=tf.int32)
            self.action_onehot = tf.one_hot(self.action, 3, dtype=tf.float32)
            self.Q = tf.reduce_sum(tf.multiply(self.Q_, self.action_onehot), axis=1)
            self.Q_next = tf.placeholder(shape=None, dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.square(self.Q_next - self.Q))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = self.optimizer.minimize(self.loss)
            self.init_op = tf.global_variables_initializer()


def fully_connected_layer(inputs, dim, activation=None, scope='fc', reuse=None,
                          init=tf.contrib.layers.xavier_initializer(), bias=True):
    with tf.variable_scope(scope, reuse=reuse):
        w_ = tf.get_variable(scope + '_W', [inputs.shape[-1], dim], initializer=init)
        outputs = tf.matmul(inputs, w_)
        if bias:
            b = tf.get_variable(scope + '_b', dim, initializer=tf.zeros_initializer())
            outputs += b
        if activation is not None:
            outputs = activation(outputs)
        return outputs


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * (1. - tau)) + (tau * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def get_epsilon(num_episode):
    return np.max([0.01, 1 - 0.0001 * num_episode])


def main():
    HER = True
    size = 100
    num_episodes = 1000
    optimisation_steps = 1000
    K = 4
    buffer_size = 10000
    tau = 0.95
    gamma = 0.99
    epsilon = 1.0
    batch_size = 64
    add_final = False

    save_model = False
    model_dir = './train'
    train = True

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    env = gym.make('guidance-collision-avoidance-single-Discrete-HER-v0')
    state_dim = env.observation_space.spaces['observation'].shape[0]
    goal_dim = 2
    modelNetwork = Model(input=state_dim, name='model')
    targetNetwork = Model(input=state_dim, name='target')
    trainables = tf.trainable_variables()
    updateOps = updateTargetGraph(trainables, tau)
    buffer = deque(maxlen=buffer_size)

    if train:
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(211)
        # plt.title('Success Rate')
        # ax.set_ylim([0, 1.])
        # ax2 = fig.add_subplot(212)
        # plt.title('Q Loss')
        # line = ax.plot(np.zeros(1), np.zeros(1), 'b-')[0]
        # line2 = ax2.plot(np.zeros(1), np.zeros(1), 'b-')[0]
        # fig.canvas.draw()
        with tf.Session() as sess:
            sess.run(modelNetwork.init_op)
            sess.run(targetNetwork.init_op)

            successes = []
            total_rewards = []
            for n in range(num_episodes):
                episode_reward = 0
                last_ob = env.reset()
                episode_experience = []  # experiences from this one episode
                episode_succeeded = False
                for t in range(size):
                    s = np.copy(last_ob['observation'])
                    g = np.copy(last_ob['desired_goal'])
                    inputs = np.concatenate([s, g], axis=-1)
                    if np.random.rand(1) < get_epsilon(n):
                        action = np.random.randint(0, 3)
                    else:
                        action = sess.run(modelNetwork.predict,
                                          feed_dict={modelNetwork.inputs: [inputs]})
                        action = action[0]
                    ob, reward, done, _ = env.step(action)
                    s_next = ob['observation']
                    episode_experience.append((s, action, reward, s_next, g))
                    last_ob = ob
                    episode_reward += reward
                    if reward == 1:
                        episode_succeeded = True
                        break
                    if done:
                        break
                successes.append(episode_succeeded)
                total_rewards.append(episode_reward)

                for t in range(len(episode_experience)):
                    s, a, r, s_n, g = episode_experience[t]
                    inputs = np.concatenate([s, g], axis=-1)
                    new_inputs = np.concatenate([s_n, g], axis=-1)
                    buffer.append([inputs, a, r, new_inputs])  # reward == 1 means terminal
                    # above is standard experience replay
                    if HER:
                        # use future HER strategy, randomly sample a future visited state as goal
                        for k in range(K):
                            future = np.random.randint(t, len(episode_experience))
                            _, _, _, g_n, _ = episode_experience[future]
                            inputs = np.concatenate([s, g_n[:2]], axis=-1)
                            new_inputs = np.concatenate([s_n, g_n[:2]], axis=-1)
                            r_n = env.compute_reward(s_n[:2], g_n[:2], _)
                            buffer.append([inputs, a, r_n, new_inputs])

                mean_loss = []
                # train network
                for k in range(optimisation_steps):
                    if len(buffer) < 1000:
                        break

                    experience = random.sample(buffer, batch_size)
                    s = np.array([ss[0] for ss in experience])
                    a = np.array([ss[1] for ss in experience])
                    r = np.array([ss[2] for ss in experience])
                    s_next = np.array([ss[3] for ss in experience])
                    Q1 = sess.run(modelNetwork.Q_, feed_dict={modelNetwork.inputs: s_next})
                    Q2 = sess.run(targetNetwork.Q_, feed_dict={targetNetwork.inputs: s_next})
                    doubleQ = Q2[:, np.argmax(Q1, axis=-1)]
                    Q_target = r + gamma * doubleQ
                    _, loss = sess.run([modelNetwork.train_op, modelNetwork.loss],
                                       feed_dict={modelNetwork.inputs: s,
                                                  modelNetwork.Q_next: Q_target,
                                                  modelNetwork.action: a})
                    mean_loss.append(loss)

                print('success rate:', np.mean(successes))
                print('loss', np.mean(mean_loss))
                updateTarget(updateOps, sess)
            # ax.relim()
            # ax.autoscale_view()
            # ax2.relim()
            # ax2.autoscale_view()
            # line.set_data(np.arange(len(success_rate)), np.array(success_rate))
            # # line.set_data(np.arange(len(total_rewards)), np.array(total_rewards))
            # line2.set_data(np.arange(len(total_loss)), np.array(total_loss))
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.pause(1e-7)

            if save_model:
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
        input('Press enter...')


    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dir, 'model.ckpt'))
        while True:
            env.reset()
            print('Initial State:\t{}'.format(env.state))
            print('Goal:\t{}'.format(env.target))
            for t in range(size):
                s = np.copy(env.state)
                g = np.copy(env.target)
                inputs = np.concatenate([s, g], axis=-1)
                action = sess.run(targetNetwork.predict, feed_dict={targetNetwork.inputs: [inputs]})
                action = action[0]
                s_next, reward = env.step(action)
                print('State at step {}: {}'.format(t, env.state))
                if reward == 0:
                    print('Success!')
                    break
            input('Press enter...')


if __name__ == '__main__':
    main()

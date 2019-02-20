"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time

from gym_guidance_collision_avoidance_single.envs import SingleAircraftHEREnv

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 2000
MAX_EP_STEPS = 1000
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.95  # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]  # you can try different target replacement strategies
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32

HER = True
K = 4

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'guidance-collision-avoidance-single-HER-v0'


###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 128, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 128, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 64, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)  # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + (1 - DONE) * self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 128
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('l2'):
                n_l2 = 128
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(net, w2) + b2)

            with tf.variable_scope('l3'):
                n_l3 = 64
                w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=init_w, trainable=trainable)
                b3 = tf.get_variable('b3', [1, n_l3], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(net, w3) + b3)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_, done):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_, DONE: done})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

goal_dim = 2
state_dim = env.observation_space.spaces['observation'].shape[0] + goal_dim
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')
with tf.name_scope('DONE'):
    DONE = tf.placeholder(tf.float32, shape=[None, 1], name='done')

sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 2)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("DDPGlogs/", sess.graph)

var = 3  # control exploration

t1 = time.time()
successes = []
for i in range(MAX_EPISODES):
    last_ob = env.reset()
    ep_reward = 0
    episode_experience = []
    episode_succeeded = False

    for t in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        s = np.copy(last_ob['observation'])
        g = np.copy(last_ob['desired_goal'])
        inputs = np.concatenate([s, g], axis=-1)
        a = actor.choose_action(inputs)
        # add randomness to action selection for exploration
        a = np.clip(np.random.normal(a, var), env.action_space.low, env.action_space.high)
        ob, r, done, info = env.step(a)
        s_ = ob['observation']
        episode_experience.append([s, a, r, s_, g, done])

        last_ob = ob
        ep_reward += r
        if r == 1:
            episode_succeeded = True
            break
        if done:
            break

        successes.append(episode_succeeded)

        if M.pointer > MEMORY_CAPACITY:
            var *= .9995  # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, state_dim + action_dim: state_dim + action_dim + 1]
            b_s_ = b_M[:, state_dim + action_dim + 1: 2 * state_dim + action_dim + 1]
            b_done = b_M[:, 2 * state_dim + action_dim + 1:]

            critic.learn(b_s, b_a, b_r, b_s_, b_done)
            actor.learn(b_s)

    # store the episode experience to replay buffer M
    for t in range(len(episode_experience)):
        s, a, r, s_n, g, done = episode_experience[t]
        inputs = np.concatenate([s, g], axis=-1)
        new_inputs = np.concatenate([s_n, g], axis=-1)
        M.store_transition(inputs, a, r, new_inputs, done)
        # above is standard Exp Replay
        if HER:
            for k in range(K):
                future = np.random.randint(t, len(episode_experience))
                _, _, _, g_n, _, _ = episode_experience[future]
                inputs = np.concatenate([s, g_n[:2]], axis=-1)
                new_inputs = np.concatenate([s_n, g_n[:2]], axis=-1)
                r_n = env.compute_reward(s_n[:2], g_n[:2], _)
                M.store_transition(inputs, a, r, new_inputs, r_n == 1)

    print('Training Episode', i)
    input('press enter...')

for i in range(100):
    last_ob = env.reset()
    done = False
    time_step = 0
    while not done and time_step < 1000:
        env.render()

        s = np.copy(last_ob['observation'])
        g = np.copy(last_ob['desired_goal'])
        inputs = np.concatenate([s, g], axis=-1)
        a = actor.choose_action(inputs)
        ob, r, done, info = env.step(a)
        last_ob = ob
        time_step += 1

print('Running time: ', time.time() - t1)

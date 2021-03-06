import numpy as np
from numpy import newaxis
import random
from collections import namedtuple, deque

# from model import QNetwork
from res_light import ResBlock
from res_light import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3  # parameter for soft update
LEARNING_RATE = 1e-4
UPDATE_EVERY = 5


class Agent():
    '''Agent interacts with the environment'''

    def __init__(self, state_size, action_size, HER=False):
        self.state_size = state_size
        self.action_size = action_size
        self.HER = HER

        # double network
        self.CA_net = QNetwork(state_size, action_size, ResBlock, [6, 6, 6]).to(device)
        self.G_net = QNetwork(6, action_size, ResBlock, [6, 6, 6]).to(device)
        # move model to either gpu or cpu
        self.target = QNetwork(state_size, action_size, ResBlock, [6, 6, 6]).to(device)
        self.optimizer = optim.Adam(self.CA_net.parameters(), lr=LEARNING_RATE)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.time_step = 0

    def step(self):
        # self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.time_step = (self.time_step + 1) % UPDATE_EVERY
        if self.time_step == 0:
            # If enough samples in memory, generate batch and learn
            if len(self.memory.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act_CA(self, state, epsilon=0.):
        '''Choose an action given state using epsilon-greedy'''
        # state = state.reshape(1, -1)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.CA_net.eval()  # change model to evaluation mode
        with torch.no_grad():  # turn off gradient descent since evaluating
            q_value = self.CA_net(state)
        self.CA_net.train()  # back to train mode

        if random.random() > epsilon:
            return np.argmax(q_value.cpu().data.numpy())  # move q_value to cpu
        else:
            return random.choice(np.arange(self.action_size))

    def act_G(self, state, epsilon=0.):
        '''Choose an action given state using epsilon-greedy'''
        # state = state.reshape(1, -1)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.G_net.eval()  # change model to evaluation mode
        with torch.no_grad():  # turn off gradient descent since evaluating
            q_value = self.G_net(state)
        self.G_net.train()  # back to train mode

        if random.random() > epsilon:
            return np.argmax(q_value.cpu().data.numpy())  # move q_value to cpu
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''learning from batch'''
        states, actions, rewards, next_states, dones = experiences
        # states = states.reshape(BATCH_SIZE, 1, -1)
        # next_states = next_states.reshape(BATCH_SIZE, 1, -1)
        # get the max predicted q value for next state
        q_target_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)
        # detach the variable from the graph using detach()
        q_target = rewards + (gamma * q_target_next * (1 - dones))
        # if episode is done q_target equals to reward only
        q_expected = self.local(states).gather(1, actions)

        # gradient step
        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()  # zero gradient if not pytorch will accmulate
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local, self.target, TAU)

    def soft_update(self, local_net, target_net, tau):
        '''soft update for parameters
        θ_target = tau*θ_local + (1 - tau)*θ_target
        '''
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def add(self, episode_experience, env):
        '''
        HER add experience after episodes terminates
        '''
        for t in range(len(episode_experience)):
            s, a, r, s_n, g, done = episode_experience[t]
            ob = np.copy(s)
            goal = np.copy(g)
            new_ob = np.copy(s_n)
            inputs = np.concatenate([ob, goal], axis=-1)
            new_inputs = np.concatenate([new_ob, goal], axis=-1)
            self.memory.add(inputs, a, r, new_inputs, done)

            if self.HER:
                for k in range(4):
                    future = np.random.randint(t, len(episode_experience))
                    _, _, _, g_n, _, _ = episode_experience[future]
                    # achieved = np.copy(new_ob[:2])
                    desired = np.copy(g_n[:2])
                    inputs = np.concatenate([ob, desired], axis=-1)
                    new_inputs = np.concatenate([new_ob, desired], axis=-1)
                    # r_n = env.compute_reward(achieved, desired, _)
                    # self.memory.add(inputs, a, r_n, new_inputs, r_n == 10)
                    r_n = env.compute_input_reward(new_inputs)
                    self.memory.add(inputs, a, r_n, new_inputs, r_n == 10 or r_n == -10)


class ReplayBuffer():
    '''Buffer stores experience'''

    def __init__(self, action_size, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.action_size = action_size
        self.batch_size = batch_size
        self.experience = namedtuple('experience', field_names=['state', 'action',
                                                                'reward', 'next_state',
                                                                'done'])

    def add(self, state, action, reward, next_state, done):
        '''add new memory'''
        memory = self.experience(state, action, reward, next_state, done)
        self.memory.append(memory)

    def sample(self):
        '''random sample a batch'''
        experiences = random.sample(self.memory, k=self.batch_size)  # return a list of experience

        # restack the list of experience into batch form
        # np.vstack stacks lists verticallly
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

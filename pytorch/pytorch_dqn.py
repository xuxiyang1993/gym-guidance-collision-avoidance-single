import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from SingleAircraftRandomEnv import SingleAircraftRandomEnv
from agent import Agent


env = SingleAircraftRandomEnv()
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
state = env.reset()

def train(n_episodes=15000, eps_start=1.0, eps_end=0.01, decay=0.999):
    # training loop
    total_rewards = []
    reward_window = deque(maxlen=100)
    epsilon = eps_start
    last_mean_reward = -np.inf # compare with reward_window to check save model or not
    for i in range(1, n_episodes+1):
        episode_timestep = 0
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # env.render()
            episode_timestep += 1
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if episode_timestep > 1000:
                break
        reward_window.append(total_reward)
        total_rewards.append(total_reward)
        epsilon = max(eps_end, decay*epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)))
        if i >= 100:
            if last_mean_reward < np.mean(reward_window):
                torch.save(agent.local.state_dict(), 'checkpoint.pth')
                print('\rEpisode {}\tAverage Score: {:.2f}\tPrevious Score: {:.2f}'.format(i, np.mean(reward_window), last_mean_reward))
                print('Model saved')
                last_mean_reward = np.mean(reward_window)
    return total_rewards

total_rewards = train()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(total_rewards)), total_rewards)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


agent.local.load_state_dict(torch.load('checkpoint.pth'))
for i in range(10):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        
env.close()
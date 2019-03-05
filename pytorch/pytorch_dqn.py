import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from SingleAircraftDiscrete9HEREnv import SingleAircraftDiscrete9HEREnv
from agent import Agent


def train(env, agent, n_episodes=30000, eps_start=1.0, eps_end=0.01, decay=0.9999,
          save_path='save_model/checkpoint.pth'):
    # training loop
    total_rewards = []
    reward_window = deque(maxlen=100)
    epsilon = eps_start
    last_mean_reward = -np.inf # compare with reward_window to check save model or not
    
    for i in range(1, n_episodes+1):
        episode_timestep = 0
        state = env.reset()
        state = np.concatenate([state['observation'],state['desired_goal']], axis=-1)
        total_reward = 0
        done = False
        
        while not done:
            # env.render()
            episode_timestep += 1
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.concatenate([next_state['observation'],next_state['desired_goal']], axis=-1)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if episode_timestep > 1000:
                break
            
        reward_window.append(total_reward)
        total_rewards.append(total_reward)
        epsilon = max(eps_end, decay * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)))
        if i >= 1000:
            if last_mean_reward < np.mean(reward_window) or i % 100 == 0:
                torch.save(agent.local.state_dict(), save_path)
                print('\rEpisode {}\tAverage Score: {:.2f}\tPrevious Score: {:.2f}'.format(i, np.mean(reward_window),
                                                                                           last_mean_reward))
                print('Model saved')
                last_mean_reward = np.mean(reward_window)
                
    torch.save(agent.local.state_dict(), save_path)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_rewards)), total_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    
def evaluate(env, agent, load_path):
    agent.local.load_state_dict(torch.load(load_path))
    for i in range(10):
        state = env.reset()
        state = np.concatenate([state['observation'],state['desired_goal']], axis=-1)
        done = False
        episode_timestep = 0
        while not done and episode_timestep < 1000:
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_state = np.concatenate([next_state['observation'],next_state['desired_goal']], axis=-1)
            state = next_state
            episode_timestep += 1

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=30000)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--save_path', '-s', type=str, default='save_model/checkpoint.pth')
    args = parser.parse_args()

    env = SingleAircraftDiscrete9HEREnv()
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    if args.train:
        train(env, agent, n_episodes=args.episodes, save_path=args.save_path)

    evaluate(env, agent, args.save_path)


if __name__ == '__main__':
    main()
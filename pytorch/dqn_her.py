import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from SingleAircraftDiscrete9HEREnv import SingleAircraftDiscrete9HEREnv
from agent_her import Agent


def train(env, agent, n_episodes=30000, eps_start=1.0, eps_end=0.01, decay=0.9999,
          save_path='save_model/checkpoint.pth'):
    # training loop
    total_rewards = []
    reward_window = deque(maxlen=100)
    epsilon = eps_start
    last_mean_reward = -np.inf  # compare with reward_window to check save model or not
    for i in range(1, n_episodes + 1):
        episode_timestep = 0
        last_ob = env.reset()
        episode_experience = []  # experience in one episode
        total_reward = 0
        done = False

        while not done:
            # env.render()
            observation = np.copy(last_ob['observation'])
            desired_goal = np.copy(last_ob['desired_goal'])
            inputs = np.concatenate([observation, desired_goal], axis=-1)
            episode_timestep += 1
            action = agent.act(inputs, epsilon)
            new_ob, reward, done, _ = env.step(action)
            new_observation = last_ob['observation']
            agent.step()
            episode_experience.append((observation, action, reward, new_observation, desired_goal, done))
            last_ob = new_ob
            total_reward += reward
            if episode_timestep > 1000:
                break

        agent.add(episode_experience, env)

        reward_window.append(total_reward)
        total_rewards.append(total_reward)
        epsilon = max(eps_end, decay * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(reward_window)))
        if i >= 1000:
            if last_mean_reward < np.mean(reward_window) or i % 100 == 0:
                torch.save(agent.local.state_dict(), 'save_model/checkpoint.pth')
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
        last_ob = env.reset()
        done = False
        episode_timestep = 0
        while not done and episode_timestep < 1000:
            state = last_ob['observation']
            goal = last_ob['desired_goal']
            inputs = np.concatenate([state, goal], axis=-1)
            action = agent.act(inputs)
            env.render()
            ob, reward, done, _ = env.step(action)
            last_ob = ob

            episode_timestep += 1

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', '-e', type=int, default=30000)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--her', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--save_path', '-s', type=str, default='save_model/checkpoint.pth')
    args = parser.parse_args()

    env = SingleAircraftDiscrete9HEREnv()
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, HER=args.her)

    if args.train:
        train(env, agent, n_episodes=args.episodes, save_path=args.save_path)

    evaluate(env, agent, args.save_path)


if __name__ == '__main__':
    main()

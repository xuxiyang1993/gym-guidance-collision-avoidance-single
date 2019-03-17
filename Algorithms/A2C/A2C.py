import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from A2C_Agent import A2C_Agent
import sys
sys.path.extend(['../../../gym-guidance-collision-avoidance-single'])
sys.path.extend(['../../Simulators'])
from Simulators.SingleAircraftDiscrete3HEREnv import SingleAircraftDiscrete3HEREnv


def train(env, agent, n_iterations, save_path):
    max_path_length = 1000
    min_timesteps_per_batch = 5000
    total_timesteps = 0
    num_path = 0

    for itr in range(n_iterations):
        print('=========== Iter %d ============' % itr)

        # collect samples (paths) until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            # loop to generate N paths/ trajectories
            steps = 0

            last_ob = env.reset()
            states, actions, rewards = [], [], []

            while True:
                # loop to generate 1 path/ trajectory

                observation = np.copy(last_ob['observation'])
                desired_goal = np.copy(last_ob['desired_goal'])
                state = np.concatenate([observation, desired_goal], axis=-1)

                states.append(state)

                # feed forward policy network and step
                action = agent.act(state)
                ob, reward, done, info = env.step(action)

                actions.append(action)
                rewards.append(reward)
                last_ob = ob

                steps += 1
                if steps > max_path_length or done:
                    break

            path = {"observation": np.array(states),
                    "reward": np.array(rewards),
                    "action": np.array(actions),
                    "info": info}
            paths.append(path)
            timesteps_this_batch += len(path['reward'])

            if timesteps_this_batch > min_timesteps_per_batch:
                # break if reaches batch size
                break

        total_timesteps += timesteps_this_batch

        agent.train(paths)

    agent.save(save_path)


def evaluate(env, agent, load_path):
    agent.load(load_path)
    for i in range(20):
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
    parser.add_argument('--iterations', '-i', type=int, default=10000)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--her', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--save_path', '-s', type=str, default='save_model/checkpoint.pth')
    args = parser.parse_args()

    env = SingleAircraftDiscrete3HEREnv()
    print('state dimension:', env.observation_space.shape)
    print('number of intruders:', env.intruder_size)
    agent = A2C_Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    if args.train:
        train(env, agent, n_iterations=args.iterations, save_path=args.save_path)

    evaluate(env, agent, args.save_path)


if __name__ == '__main__':
    main()

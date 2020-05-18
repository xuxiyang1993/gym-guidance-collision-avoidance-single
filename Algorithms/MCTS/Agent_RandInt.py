import argparse
import time

import sys
sys.path.extend(['../../../gym-guidance-collision-avoidance-single'])
sys.path.extend(['../../Simulators'])
from nodes_single import SingleAircraftNode, SingleAircraftState
from search_single import MCTS
from Simulators.SingleAircraftMCTSRandIntruderEnv import SingleAircraftEnv


def run_experiment(env, no_episodes, no_simulations, search_depth):
    episode = 0
    epi_returns = []
    running_time_list = []
    conflicts = []

    while episode < no_episodes:
        # at the beginning of each episode, set done to False, set time step in this episode to 0
        # set reward to 0, reset the environment
        episode += 1
        done = False
        episode_time_step = -1
        episode_reward = 0
        last_observation = env.reset()
        action = None
        info = 'early termination'

        while not done:
            # at each time step
            env.render()
            episode_time_step += 1

            if episode_time_step % 5 == 0:

                time_before = int(round(time.time() * 1000))
                state = SingleAircraftState(state=last_observation)
                root = SingleAircraftNode(state=state)
                mcts = MCTS(root)
                best_node = mcts.best_action(no_simulations, search_depth)
                action = best_node.state.prev_action
                time_after = int(round(time.time() * 1000))
                running_time_list.append(time_after - time_before)

            observation, reward, done, info = env.step(action)

            episode_reward += reward
            last_observation = observation

        # print training information for each training episode
        epi_returns.append(info)
        conflicts.append(env.no_conflict)
        print('Training Episode:', episode)

    env.close()
    print('----------------------------------------')
    # print('intruders: ', 80)
    print('search depth: ', search_depth)
    print('simulation: ', no_simulations)
    print('time: ', sum(running_time_list) / float(len(running_time_list)))
    print('NMAC prob: ', epi_returns.count('n') / no_episodes)
    print('goal prob: ', epi_returns.count('g') / no_episodes)
    print('average conflicts: ', sum(conflicts) / no_episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_episodes', '-e', type=int, default=10)
    parser.add_argument('--no_simulations', '-s', type=int, default=100)
    parser.add_argument('--search_depth', '-d', type=int, default=3)
    parser.add_argument('--config_path', '-cf', type=str, default='config/config_file.ini')
    args = parser.parse_args()

    env = SingleAircraftEnv()
    run_experiment(env, args.no_episodes, args.no_simulations, args.search_depth)


if __name__ == '__main__':
    main()

**Status:** Active (under active development, breaking changes may occur)

Reference:  https://www.researchgate.net/publication/327174624_Autonomous_On-Demand_Free_Flight_Operations_in_Urban_Air_Mobility_using_Monte_Carlo_Tree_Search

# gym-guidance-collision-avoidance-single

Control a single aircraft (yellow aircraft) to reach the goal position (green star) while avoiding conflicts with other intruder aircraft (red aircraft).

Parameter of the environments can be found in `config.py`.

## SingleAircraftEnv
The action space is discrete: -1, 0, 1 for change of heading and -1, 0, 1 for the throttle.

## SingleAircraft2Env
The action space is continuous: [-1, 1] for change of heading and [-1, 1] for the throttle.

## SingleAircraftStackEnv
The input state is the whole image of the map, with four most recent frames stacked together. Action space is discrete.

## SingleAircraftHEREnv
The environment designed to implement Hindsight Experience Replay algorithm. The observation space is of the following form:
```
OrderedDict([('achieved_goal', Box),
             ('desired_goal', Box),
             ('observation', Box)])
```

The action space is continuous.

## SingleAircraftDiscrete3HEREnv, SingleAircraftDiscrete9HEREnv
The environment designed to implement Hindsight Experience Replay algorithm with 3/9 discrete actions.

<!--
## Installation

```bash
cd gym-guidance-collision-avoidance-single
pip install -e .
```
-->

# OpenAI baselines

You can use baselines algorithms from OpenAI to solve this problem. Simply run the following:

For discrete action space
```base
cd baselines-master
python -m baselines.run --env=guidance-collision-avoidance-single-v0 --alg=deepq
```
For continuous action space
```base
cd baselines-master
python -m baselines.run --env=guidance-collision-avoidance-single-continuous-action-v0 --alg=ddpg
```

Optional arguments:

`--save_path` the path where you want to save the model

`--load_path` the path where you want to load the model

`--num_timesteps` total time steps the agent will be trained

More optional can be found in the documentation of OpenAI baselines repository.

# MCTS

Use Monte Carlo Tree Search Algorithm to solve this problem. Details can be found at the above reference. Run

```bash
cd Algorithms/MCTS
python Agent.py
```

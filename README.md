**Status:** Under development, can **NOT** guarantee bug-free

Reference:  https://www.researchgate.net/publication/327174624_Autonomous_On-Demand_Free_Flight_Operations_in_Urban_Air_Mobility_using_Monte_Carlo_Tree_Search

# gym-guidance-collision-avoidance-single

Control a single aircraft to reach the goal position while avoiding conflicts with other intruder aircraft.
This repository contrains two environments.

## Guidance and Collision Avoidance with discrete action space
For this env, the action is to take -1, 0, 1 for change of heading and -1, 0, 1 for the throttle.

## Guidance and Collision Avoidance with continuous action space
For this env, the action is continuous with range -1 to 1 for both the change of heading and throttle.

Parameter of the above two environments can be found in `config.py`.

## Installation

```bash
cd gym-guidance-collision-avoidance-single
pip install -e .
```

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
python Agent.py
```

**Status:** Under development, can **NOT** guarantee bug-free

Reference:  https://www.researchgate.net/publication/327174624_Autonomous_On-Demand_Free_Flight_Operations_in_Urban_Air_Mobility_using_Monte_Carlo_Tree_Search

# gym-guidance-collision-avoidance-single

Control a single aircraft to reach the goal position while avoiding conflicts with other intruder aircraft.

Parameter of this simulator can be found in `configs/config_file.ini`.

## Installation

```bash
cd gym-guidance-collision-avoidance-single
pip install -e .
```

# MCTS

Use Monte Carlo Tree Search Algorithm to solve this problem. Details can be found at the above reference. Run

```bash
python Agent.py
```

(there is a module import error)

`env.render()` is currently under working.

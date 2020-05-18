import copy
import itertools
import math
import numpy as np

from config_single import Config
from common import MCTSNode, MCTSState


class SingleAircraftState(MCTSState):
    def __init__(self, state, hit_wall=False, conflict=False, reach_goal=False, prev_action=None, depth=0):
        MCTSState.__init__(self, state)
        self.hit_wall = hit_wall
        self.conflict = conflict
        self.reach_goal = reach_goal
        self.prev_action = prev_action
        self.depth = depth
        self.config = Config()
        self.G = self.config.G
        self.scale = self.config.scale

        self.nearest_x = -1
        self.nearest_y = -1

    def reward(self):
        if self.hit_wall or self.conflict:
            r = 0
        elif self.reach_goal:
            r = 1
        else:
            r = 1 - self.dist_goal() / 1200.0  # - self.dist_intruder() / 1200
        return r

    def is_terminal_state(self, search_depth):
        if self.reach_goal or self.conflict or self.hit_wall or self.depth == search_depth:
            return True
        return False

    def move(self, action):
        # state: [intruder: x, y, vx, yv]
        # [ownship: x, y, vx, vy, v, psi, phi]
        # [goal: x, y]
        state = copy.deepcopy(self.state)
        hit_wall = False
        conflict = False
        reach_goal = False
        near_intruders_size = (len(state) - 8) // 6

        d_heading = (action[0] - 1) * self.config.d_heading
        acceleration = (action[1] - 1) * self.config.d_speed

        for _ in range(self.config.simulate_frame):
            # update intruder
            for i in range(near_intruders_size):
                # the intruder aircraft has some uncertainty
                vx = state[6 * i + 2]
                vy = state[6 * i + 3]
                speed = state[6 * i + 4]
                heading = state[6 * i + 5]

                state[6 * i + 0] += vx + np.random.normal(0, self.config.position_sigma)
                state[6 * i + 1] += vy + np.random.normal(0, self.config.position_sigma)

                if np.random.random() < 0.1:
                    heading += math.radians(np.random.uniform(-10, 10))
                    vx = speed * math.cos(heading)
                    vy = speed * math.sin(heading)
                    state[6 * i + 2] = vx
                    state[6 * i + 3] = vy
                    state[6 * i + 5] = heading


            state[-4] += acceleration  # update speed according to acceleration
            state[-4] = max(self.config.min_speed, min(state[-4], self.config.max_speed))
            state[-4] += np.random.normal(0, self.config.speed_sigma)
            speed = state[-4]
            state[-3] += d_heading  # update bank angle according to delta bank
            state[-3] += np.random.normal(0, self.config.heading_sigma)
            heading = state[-3]

            ownvx = speed * math.cos(heading)
            ownvy = speed * math.sin(heading)
            state[-8] += ownvx
            ownx = state[-8]
            state[-7] += ownvy
            owny = state[-7]

            state[-6] = ownvx
            state[-5] = ownvy

            goalx = state[-2]
            goaly = state[-1]

            if not 0 < ownx < self.config.window_width or not 0 < owny < self.config.window_height:
                hit_wall = True
                break

            for i in range(near_intruders_size):
                x = state[4 * i + 0]
                y = state[4 * i + 1]
                if self.metric(x, y, ownx, owny) < self.config.minimum_separation:
                    conflict = True
                    # who cares next state?
                    break

            if conflict:
                break

            if self.metric(ownx, owny, goalx, goaly) < self.config.minimum_separation:
                reach_goal = True
                # Terminate!
                break

        return SingleAircraftState(state, hit_wall, conflict, reach_goal, action, self.depth + 1)

    def get_legal_actions(self):
        return list(itertools.product(range(3), repeat=2))

    def dist_goal(self):
        dx = self.ownx - self.goalx
        dy = self.owny - self.goaly
        return np.sqrt(dx**2 + dy**2)

    def dist_intruder(self):
        distance = 5000
        near_intruders_size = (len(self.state) - 9) // 4
        for i in range(near_intruders_size):
            intrux = self.state[4 * i + 0]
            intruy = self.state[4 * i + 1]
            current_dist = self.metric(intrux, intruy, self.ownx, self.owny)
            if current_dist < distance:
                distance = current_dist
                self.nearest_x = intrux
                self.nearest_y = intruy
        return distance

    def metric(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return np.sqrt(dx**2 + dy**2)

    @property
    def ownx(self):
        return self.state[-8]

    @property
    def owny(self):
        return self.state[-7]

    @property
    def own_vx(self):
        return self.state[-6]

    @property
    def own_vy(self):
        return self.state[-5]

    @property
    def own_speed(self):
        return self.state[-4]

    @property
    def own_heading(self):
        return self.state[-3]

    @property
    def goalx(self):
        return self.state[-2]

    @property
    def goaly(self):
        return self.state[-1]

    def __repr__(self):
        s = 'prev action: %s, own: (%f, %f), bank: %d, goal: (%f, %f), dist goal: %f, dist intruder: %f,' \
            'nearest intruder: (%.2f, %.2f), depth: %d' \
            % (self.prev_action,
               self.ownx,
               self.owny,
               self.own_bank,
               self.goalx,
               self.goaly,
               self.dist_goal(),
               self.dist_intruder(),
               self.nearest_x,
               self.nearest_y,
               self.depth)
        return s


class SingleAircraftNode(MCTSNode):
    def __init__(self, state: SingleAircraftState, parent=None):
        MCTSNode.__init__(self, parent)
        self.state = state

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = SingleAircraftNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self, search_depth):
        return self.state.is_terminal_state(search_depth)

    def rollout(self, search_depth):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal_state(search_depth):
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.reward()

    def backpropagate(self, result):
        self.n += 1.
        self.q += result
        if self.parent:
            self.parent.backpropagate(result)

    def __repr__(self):
        s = 'Node: children: %d; visits: %d; reward: %.4f; p_action: %s, state: (%.2f, %.2f); ' \
            'bank: %d, goal: (%.2f, %.2f), dist: %.2f, dist2: %.2f, nearest: (%.2f, %.2f)' \
            % (len(self.children),
               self.n,
               self.q / (self.n + 1e-2),
               self.state.prev_action,
               self.state.ownx,
               self.state.owny,
               self.state.own_bank,
               self.state.goalx,
               self.state.goaly,
               self.state.dist_goal(),
               self.state.dist_intruder(),
               self.state.nearest_x,
               self.state.nearest_y)

        return s


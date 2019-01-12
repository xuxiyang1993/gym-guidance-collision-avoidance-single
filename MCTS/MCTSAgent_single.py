import copy
import hashlib
import math
import os
import itertools

import numpy as np
from configparser import ConfigParser


class State:
    def __init__(self, state, collide_wall=False, conflict_intruder=False,
                 reach_goal=False, prev_action=None, depth=0):
        self.state = state
        self.collide_wall = collide_wall
        self.conflict_intruder = conflict_intruder
        self.reach_goal = reach_goal
        self.prev_action = prev_action
        self.depth = depth

    def next_state(self, a):
        return PROCEED(self.state, a, self.depth)

    def terminal(self):
        if self.reach_goal or self.conflict_intruder or self.collide_wall or self.depth == search_depth:
            return True
        return False

    def reward(self):
        if self.collide_wall:
            r = 0
        elif self.conflict_intruder:
            r = 0
        elif self.reach_goal:
            r = +1
        else:
            r = 1 - self.dist_goal() / 1200.0  # - self.dist_intruder() / 1200
        return r

    @property
    def ownx(self):
        return self.state[-9]

    @property
    def owny(self):
        return self.state[-8]

    @property
    def own_bank(self):
        return self.state[-3]

    @property
    def goalx(self):
        return self.state[-2]

    @property
    def goaly(self):
        return self.state[-1]

    def dist_goal(self):
        return metric(self.ownx, self.owny, self.goalx, self.goaly)

    def dist_intruder(self):
        distance = 5000
        near_intruders_size = (len(self.state) - 9) // 4
        for i in range(near_intruders_size):
            intrux = self.state[4 * i + 0]
            intruy = self.state[4 * i + 1]
            current_dist = metric(intrux, intruy, self.ownx, self.owny)
            if current_dist < distance:
                distance = current_dist
                self.nearest_x = intrux
                self.nearest_y = intruy

        return distance

    def __hash__(self):
        this_str = str(self.state) + str(self.collide_wall) + str(self.conflict_intruder) + str(self.reach_goal)
        return int(hashlib.md5(this_str.encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = 'prev action: %d, own: (%f, %f), bank: %d, goal: (%f, %f), dist goal: %f, dist intruder: %f,' \
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


class Node:
    def __init__(self, state, parent=None):
        self.visits = 0
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent
        self.untried_action = list(itertools.product(range(3), repeat=2))

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == 9:
            return True
        return False

    def __repr__(self):
        s = "Node: children: %d; visits: %d; reward: %.4f; p_action: %s, state: (%.2f, %.2f); " \
            "bank: %d, goal: (%.2f, %.2f), dist: %.2f, dist2: %.2f, nearest: (%.2f, %.2f)" \
            % (len(self.children),
               self.visits,
               self.reward / self.visits,
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


class SingleMCTSAgent:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        parser = ConfigParser(os.environ)
        parser.read(self.config_path)
        self.C = parser.getfloat('algorithm', 'C')

    def UCTSEARCH(self, simulations, root):
        for _ in range(int(simulations)):
            # if iter % 10000 == 9999:
            #     print("simulation: %d" % iter)
            #     print(root)
            front = self.TREEPOLICY(root)
            reward = self.DEFAULTPOLICY(front.state)
            self.BACKUP(front, reward)
            # print(front.state)
            # print(reward)
            # print('---------------------------------------------')
        return self.BESTCHILD(root, 0)

    def TREEPOLICY(self, node):
        # a hack to force 'exploitation' in a game where there are many options,
        # and you may never/not want to fully expand first
        while node.state.terminal() == False:
            # if len(node.children) == 0:
            #     return EXPAND(node)
            # elif random.uniform(0, 1) < .5:
            #     node = BESTCHILD(node, SCALAR)
            # else:
            #     if node.fully_expanded() == False:
            #         return EXPAND(node)
            #     else:
            #         node = BESTCHILD(node, SCALAR)
            if not node.fully_expanded():
                return self.EXPAND(node)
            else:
                node = self.BESTCHILD(node, self.C)
        return node

    def EXPAND(self, node):
        # tried_children = [c.state for c in node.children]
        # new_state = node.state.next_state()
        # while new_state in tried_children:
        #     new_state = node.state.next_state()
        # node.add_child(new_state)
        # return node.children[-1]
        random_index = np.random.randint(0, len(node.untried_action))
        random_action = node.untried_action[random_index]
        node.untried_action.remove(random_action)
        new_state = node.state.next_state(random_action)
        node.add_child(new_state)
        return node.children[-1]

    # current this uses the most vanilla MCTS formula it is worth
    # experimenting with THRESHOLD ASCENT (TAGS)
    def BESTCHILD(self, node, scalar):
        bestscore = -10
        bestchildren = []
        for c in node.children:
            exploit = c.reward / c.visits
            explore = math.sqrt(2 * math.log(node.visits) / float(c.visits))
            score = exploit + scalar * explore
            if score == bestscore:
                bestchildren.append(c)
            if score > bestscore:
                bestchildren = [c]
                bestscore = score
        if len(bestchildren) == 0:
            print("OOPS: no best child found, probably fatal")
        if len(bestchildren) > 1:
            for node in bestchildren:
                if node.state.prev_action[0] == 1:
                    return node
        if scalar == 0:
            r = np.random.choice(bestchildren)
            return r
        return np.random.choice(bestchildren)

    def DEFAULTPOLICY(self, state):
        while state.terminal() == False:
            random_action = np.random.randint(0, 3, (2,))
            state = state.next_state(random_action)
            # print(state)
        return state.reward()

    def BACKUP(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
        return

    def PROCEED(self, s, a, d):

        return State()

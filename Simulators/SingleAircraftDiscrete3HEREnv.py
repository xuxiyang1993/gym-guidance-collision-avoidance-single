import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from config import Config

__author__ = "Xuxi Yang <xuxiyang@iastate.edu>"


class SingleAircraftDiscrete3HEREnv(gym.GoalEnv):
    """
    This is the airspace simulator where we can control single aircraft (yellow aircraft)
    to reach the goal position (green star) while avoiding conflicts with other intruder aircraft (red aircraft).
    **STATE:**
    The state consists all the information needed for the ownship to choose an optimal action:
    position, velocity of intruder aircraft
    position, velocity, speed, heading angle, bank angle
    position of the goal
    In the beginning of each episode, the ownship starts in the bottom right corner of the map and the heading angle is
    pointing directly to the center of the map.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 for the change of bank angle and +1, 0, -1 for the change of acceleration
    """

    def __init__(self):
        self.load_config()
        self.state = None
        self.viewer = None

        obs = self.reset()
        # build observation space and action space
        '''
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float)
        '''
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(obs['observation'].shape[0]+2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.position_range = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height]),
            dtype=np.float32)

        self.seed(2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_config(self):
        # input dim
        self.window_width = Config.window_width
        self.window_height = Config.window_height
        self.intruder_size = Config.intruder_size
        self.EPISODES = Config.EPISODES
        self.G = Config.G
        self.tick = Config.tick
        self.scale = Config.scale
        self.minimum_separation = Config.minimum_separation
        self.NMAC_dist = Config.NMAC_dist
        self.horizon_dist = Config.horizon_dist
        self.initial_min_dist = Config.initial_min_dist
        self.goal_radius = Config.goal_radius
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed

    def reset(self):
        # ownship = recordtype('ownship', ['position', 'velocity', 'speed', 'heading', 'bank'])
        # intruder = recordtype('intruder', ['id', 'position', 'velocity'])
        # goal = recordtype('goal', ['position'])

        # initiate ownship to control
        self.drone = Ownship(
            position=self.random_pos(),
            speed=self.random_speed(),
            heading=self.random_heading()
        )

        # randomly generate intruder aircraft and store them in a list
        self.intruder_list = []
        for _ in range(self.intruder_size):
            intruder = Aircraft(
                position=self.random_pos(),
                speed=self.random_speed(),
                heading=self.random_heading(),
            )
            # new intruder aircraft can appear too close to ownship
            while dist(self.drone, intruder) < self.initial_min_dist:
                intruder.position = self.random_pos()

            self.intruder_list.append(intruder)

        # generate a random goal
        self.goal = Goal(position=self.random_pos())

        # reset the number of conflicts to 0
        self.no_conflict = 0

        return self._get_ob()

    def _get_ob(self):
        # state contains pos, vel for all intruder aircraft
        # pos, vel, speed, heading for ownship
        # goal pos
        def normalize_velocity(velocity):
            translation = velocity + self.max_speed
            return translation / (self.max_speed * 2)

        s = []

        # ownship information #####################################
        for i in range(1):
            # (x, y, vx, vy, speed, heading)
            s.append(self.drone.position[0] / Config.window_width)
            s.append(self.drone.position[1] / Config.window_height)
            s.append(normalize_velocity(self.drone.velocity[0]))
            s.append(normalize_velocity(self.drone.velocity[1]))
            # s.append((self.drone.speed - Config.min_speed) / (Config.max_speed - Config.min_speed))
            # s.append(self.drone.heading / (2 * math.pi))
            dx, dy = self.goal.position - self.drone.position
            desired_heading = math.atan2(dy, dx) / (2 * math.pi)
            s.append(desired_heading)
        # #########################################################

        # n nearest aircraft ######################################
        dist_list = []
        for aircraft in self.intruder_list:
            dist_list.append(dist(aircraft, self.drone))

        dist_array = np.array(dist_list)

        ind = np.argpartition(dist_array, Config.n)[0:Config.n]
        closest_ind = ind[np.argsort(dist_array[ind])]

        for i in closest_ind:
            # (x, y, vx, vy)
            s.append(self.intruder_list[i].position[0] / Config.window_width)
            s.append(self.intruder_list[i].position[1] / Config.window_height)
            s.append(normalize_velocity(self.intruder_list[i].velocity[0]))
            s.append(normalize_velocity(self.intruder_list[i].velocity[1]))
            s.append(dist(self.intruder_list[i], self.drone) / Config.diagonal)
        # #########################################################

        # all intruder aircraft ###################################
        # for aircraft in self.intruder_list:
        #     # (x, y, vx, vy)
        #     s.append(aircraft.position[0] / Config.window_width)
        #     s.append(aircraft.position[1] / Config.window_height)
        #     s.append(normalize_velocity(aircraft.velocity[0]))
        #     s.append(normalize_velocity(aircraft.velocity[1]))
        # #########################################################

        achieved_goal = np.array([self.drone.position[0] / Config.window_width,
                                  self.drone.position[1] / Config.window_height])

        desired_goal = np.array([self.goal.position[0] / Config.window_width,
                                 self.goal.position[1] / Config.window_height])

        return {
            'observation': np.array(s),
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }

    def step(self, action):

        # next state of ownship
        self.drone.step(action)

        reward, terminal, info = self._terminal_reward()

        return self._get_ob(), reward, terminal, {'result': info}

    def _terminal_reward(self):

        # step the intruder aircraft
        conflict = False
        # for each aircraft
        for idx in range(self.intruder_size):
            intruder = self.intruder_list[idx]
            intruder.position += intruder.velocity
            dist_intruder = dist(self.drone, intruder)
            # if this intruder out of map
            if not self.position_range.contains(intruder.position):
                self.intruder_list[idx] = self.reset_intruder()

            # if there is a conflict
            if dist_intruder < self.minimum_separation:
                conflict = True
                # if conflict status transition from False to True, increase number of conflicts by 1
                # if conflict status is True, monitor when this conflict status will be escaped
                if intruder.conflict == False:
                    self.no_conflict += 1
                    intruder.conflict = True
                else:
                    if not dist_intruder < self.minimum_separation:
                        intruder.conflict = False

                # if there is a near-mid-air-collision
                if dist_intruder < self.NMAC_dist:
                    return Config.NMAC_penalty, True, 'n'  # NMAC

        # if there is conflict
        if conflict:
            return Config.conflict_penalty, False, 'c'  # conflict

        # if ownship out of map
        if Config.sparse_reward:
            if not self.position_range.contains(self.drone.position):
                return Config.wall_penalty, True, 'w'  # out-of-map
        
        # if ownship reaches goal
        if dist(self.drone, self.goal) < self.goal_radius:
            return Config.goal_reward, True, 'g'  # goal

        if Config.sparse_reward:
            return Config.step_penalty, False, ''
        else:
            return -dist(self.drone, self.goal) / 1200, False, ''
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = self.unnormalize_position(achieved_goal)
        desired_goal = self.unnormalize_position(desired_goal)
        d = np.linalg.norm((achieved_goal - desired_goal), axis=-1)
        if Config.sparse_reward:
            if d < self.goal_radius:
                return Config.goal_reward
            else:
                return Config.step_penalty
        else:
            if d < self.goal_radius:
                return Config.goal_reward
            else:
                return - d / 1200

    def compute_input_reward(self, new_inputs):
        ownx = new_inputs[0] * Config.window_width
        owny = new_inputs[1] * Config.window_height
        gx = new_inputs[-2] * Config.window_width
        gy = new_inputs[-1] * Config.window_height

        dist_goal = metric(ownx, owny, gx, gy)
        
        if Config.intruder_size != 0:
            for idx in range(Config.n):
                intrux = new_inputs[idx * 4 + 4] * Config.window_width
                intruy = new_inputs[idx * 4 + 5] * Config.window_height
    
                dist_intruder = metric(ownx, owny, intrux, intruy)
    
                # if there is a conflict
                if dist_intruder < self.minimum_separation:
                    reward = Config.conflict_penalty
    
                    # if there is a near-mid-air-collision
                    if dist_intruder < self.NMAC_dist:
                        reward = Config.NMAC_penalty
    
                    return reward

        # if ownship reaches goal
        if dist_goal < self.goal_radius:
            return Config.goal_reward

        if Config.sparse_reward:
            return Config.step_penalty
        else:
            return -dist_goal / 1200

    def unnormalize_position(self, position):
        #print('position is: ', position)
        position[0] = position[0] * Config.window_width
        position[1] = position[1] * Config.window_height
        return position
        
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        if self.drone is None:
            return None

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # draw ownship
        ownship_img = rendering.Image(os.path.join(__location__, 'images/aircraft.png'), 32, 32)
        jtransform = rendering.Transform(rotation=self.drone.heading - math.pi / 2, translation=self.drone.position)
        ownship_img.add_attr(jtransform)
        ownship_img.set_color(255, 241, 4)  # set it to yellow
        self.viewer.onetime_geoms.append(ownship_img)

        # draw goal
        goal_img = rendering.Image(os.path.join(__location__, 'images/goal.png'), 32, 32)
        jtransform = rendering.Transform(rotation=0, translation=self.goal.position)
        goal_img.add_attr(jtransform)
        goal_img.set_color(15, 210, 81)  # green
        self.viewer.onetime_geoms.append(goal_img)

        # draw intruders
        for aircraft in self.intruder_list:
            intruder_img = rendering.Image(os.path.join(__location__, 'images/intruder.png'), 32, 32)
            jtransform = rendering.Transform(rotation=aircraft.heading - math.pi / 2, translation=aircraft.position)
            intruder_img.add_attr(jtransform)
            intruder_img.set_color(237, 26, 32)  # red color
            self.viewer.onetime_geoms.append(intruder_img)

        return self.viewer.render(return_rgb_array=False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # reset pos, vel, heading of this aircraft
    def reset_intruder(self):
        intruder = Aircraft(
            position=self.random_pos(),
            speed=self.random_speed(),
            heading=self.random_heading(),
        )
        while dist(self.drone, intruder) < self.initial_min_dist:
            intruder.position = self.random_pos()

        return intruder

    def random_pos(self):
        return np.random.uniform(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height])
        )

    def random_speed(self):
        return np.random.uniform(low=self.min_speed, high=self.max_speed)

    def random_heading(self):
        return np.random.uniform(low=0, high=2 * math.pi)

    def build_observation_space(self):
        s = spaces.Dict({
            'own_x': spaces.Box(low=0, high=self.window_width, dtype=np.float32),
            'own_y': spaces.Box(low=0, high=self.window_height, dtype=np.float32),
            'pos_x': spaces.Box(low=0, high=self.window_width, dtype=np.float32),
            'pos_y': spaces.Box(low=0, high=self.window_height, dtype=np.float32),
            'heading': spaces.Box(low=0, high=2 * math.pi, dtype=np.float32),
            'speed': spaces.Box(low=self.min_speed, high=self.max_speed, dtype=np.float32),
        })
        return s


class Goal:
    def __init__(self, position):
        self.position = position


class Aircraft:
    def __init__(self, position, speed, heading):
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.heading = heading  # rad
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy], dtype=np.float32)

        self.conflict = False  # track if this aircraft is in conflict with ownship


class Ownship(Aircraft):
    def __init__(self, position, speed, heading):
        Aircraft.__init__(self, position, speed, heading)
        self.load_config()

    def load_config(self):
        self.G = Config.G
        self.scale = Config.scale
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed
        self.d_speed = Config.d_speed
        self.speed_sigma = Config.speed_sigma
        self.position_sigma = Config.position_sigma

        self.d_heading = Config.d_heading
        self.heading_sigma = Config.heading_sigma

    def step(self, a):
        self.heading += self.d_heading * (a - 1)
        self.heading += np.random.normal(0, self.heading_sigma)
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))  # project to range
        self.speed += np.random.normal(0, self.speed_sigma)

        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy])
        self.position += self.velocity


def dist(object1, object2):
    return np.linalg.norm(object1.position - object2.position)


def metric(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx ** 2 + dy ** 2)

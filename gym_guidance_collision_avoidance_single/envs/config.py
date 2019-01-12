import os
from configparser import ConfigParser


class Config:
    def __init__(self):
        self.config_path = 'config/config_file.ini'
        parser = ConfigParser(os.environ)
        parser.read(self.config_path)
        # input dim
        self.window_width = parser.getint('simulator', 'width')
        self.window_height = parser.getint('simulator', 'height')
        self.intruder_size = parser.getint('simulator', 'intruder_size')
        self.EPISODES = parser.getint('simulator', 'EPISODES')
        self.G = parser.getfloat('simulator', 'G')
        self.tick = parser.getint('simulator', 'tick')
        self.scale = parser.getint('simulator', 'SCALE')
        self.minimum_separation = parser.getint('simulator', 'minimum_separation')/self.scale
        self.NMAC_dist = parser.getint('simulator', 'NMAC_dist')/self.scale
        self.horizon_dist = parser.getint('simulator', 'horizon_dist')/self.scale
        self.initial_min_dist = parser.getint('simulator', 'initial_min_dist')/self.scale
        self.goal_radius = parser.getint('simulator', 'goal_radius')/self.scale
        self.min_speed = parser.getint('aircraft_model', 'min_speed')/self.scale
        self.max_speed = parser.getint('aircraft_model', 'max_speed')/self.scale
        self.d_speed = parser.getint('aircraft_model', 'd_speed') / self.scale
        self.speed_sigma = parser.getint('uncertainty', 'speed_sigma') / self.scale
        self.position_sigma = parser.getint('uncertainty', 'position_sigma') / self.scale

        self.min_bank = parser.getint('aircraft_model', 'min_bank')
        self.max_bank = parser.getint('aircraft_model', 'max_bank')
        self.d_bank = parser.getint('aircraft_model', 'd_bank')
        self.bank_sigma = parser.getint('uncertainty', 'bank_sigma')

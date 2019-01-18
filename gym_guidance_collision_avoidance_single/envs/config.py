import os
from configparser import ConfigParser


class Config:
    config_path = 'config/config_file.ini'
    parser = ConfigParser(os.environ)
    parser.read(config_path)
    # input dim
    window_width = parser.getint('simulator', 'width')
    window_height = parser.getint('simulator', 'height')
    intruder_size = parser.getint('simulator', 'intruder_size')
    EPISODES = parser.getint('simulator', 'EPISODES')
    G = parser.getfloat('simulator', 'G')
    tick = parser.getint('simulator', 'tick')
    scale = parser.getint('simulator', 'SCALE')
    minimum_separation = parser.getint('simulator', 'minimum_separation')/scale
    NMAC_dist = parser.getint('simulator', 'NMAC_dist')/scale
    horizon_dist = parser.getint('simulator', 'horizon_dist')/scale
    initial_min_dist = parser.getint('simulator', 'initial_min_dist')/scale
    goal_radius = parser.getint('simulator', 'goal_radius')/scale
    min_speed = parser.getint('aircraft_model', 'min_speed')/scale
    max_speed = parser.getint('aircraft_model', 'max_speed')/scale
    d_speed = parser.getint('aircraft_model', 'd_speed') / scale
    speed_sigma = parser.getint('uncertainty', 'speed_sigma') / scale
    position_sigma = parser.getint('uncertainty', 'position_sigma') / scale

    min_bank = parser.getint('aircraft_model', 'min_bank')
    max_bank = parser.getint('aircraft_model', 'max_bank')
    d_bank = parser.getint('aircraft_model', 'd_bank')
    bank_sigma = parser.getint('uncertainty', 'bank_sigma')

    simulate_frame = parser.getint('algorithm', 'simulate_frame')

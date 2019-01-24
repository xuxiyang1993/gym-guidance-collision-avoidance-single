import math


class Config:
    # input dim
    window_width = 800
    window_height = 800
    intruder_size = 10
    EPISODES = 1000
    G = 9.8
    tick = 30
    scale = 30

    # distance param
    minimum_separation = 555 / scale
    NMAC_dist = 150 / scale
    horizon_dist = 4000 / scale
    initial_min_dist = 3000 / scale
    goal_radius = 600 / scale

    # speed
    min_speed = 50 / scale
    max_speed = 80 / scale
    d_speed = 5 / scale
    speed_sigma = 2 / scale
    position_sigma = 10 / scale

    # heading in rad TBD
    d_heading = math.radians(5)
    heading_sigma = math.radians(5)

    # bank
    min_bank = -25
    max_bank = 25
    d_bank = 5
    bank_sigma = 4

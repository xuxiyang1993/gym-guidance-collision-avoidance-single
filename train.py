import gym

from gym_guidance_collision_avoidance_single.envs import SingleAircraft2Env

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

env = gym.make('guidance-collision-avoidance-single-continuous-action-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = DDPG(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

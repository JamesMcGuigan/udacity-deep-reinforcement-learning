# Source: https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
import os
from collections import deque

import gym
import numpy as np
from gym.vector.utils import spaces

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from unityagents import UnityEnvironment


# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)
from src.libs.env import get_env_state_action_agents_size


class ReacherOneUnityEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10):
        super().__init__()
        os.chdir(os.path.dirname(__file__))
        self.env = UnityEnvironment(file_name='../unity/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86')
        state_size, action_size, num_agents = get_env_state_action_agents_size(self.env)
        self.scores_window = []

        # Define action and observation space
        # They must be gym.spaces objects
        # self.action_space = spaces.Discrete(action_size)
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(action_size,),
            dtype=np.float32
        )

        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(state_size,),
            dtype=np.float32
        )

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        response    = self.env.reset()
        response    = list(response.values())[0]
        observation = response.vector_observations[0]
        reward      = response.rewards[0]
        done        = response.local_done[0]
        return observation


    def step(self, action):
        response    = self.env.step(action)
        response    = list(response.values())[0]
        observation = response.vector_observations[0]
        reward      = response.rewards[0]
        done        = response.local_done[0]
        info        = {}

        self.scores_window.append(reward)
        if len(self.scores_window) % 1000 == 0:
            scores = self.scores_window[-1000:]
            print(f'step {len(self.scores_window)} | non_zero {np.count_nonzero(scores):2d} | min {min(scores):.3f} | mean {np.mean(scores):.3f} | max {max(scores):.3f}')

        return observation, reward, done, info

    def render(self, mode='console'):
        # raise NotImplementedError()
        pass

    def close(self):
        return self.env.close()


env   = ReacherOneUnityEnv()
# model = A2C("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ReacherOneUnity")

del model # remove to demonstrate saving and loading
model = A2C.load("ReacherOneUnity")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

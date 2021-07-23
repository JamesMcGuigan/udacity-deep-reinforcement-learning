# Source: https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html

import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from unityagents import UnityEnvironment


# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)
env = UnityEnvironment(file_name='./unity/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')


model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

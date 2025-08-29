# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import gymnasium as gym
env= gym.make('MountainCarContinuous-v0',render_mode='rgb_array')
env_name="MountainCarContinuous"

Algo_name="Random_test"

# %%
import matplotlib.pyplot as plt
from matplotlib import animation , rc
from IPython import display
# %matplotlib inline

# %%
import ffmpegio
import numpy as np
import os

# %%
imgs_dir = f"imgs/{env_name}/{Algo_name}"

# %%
# Observation and action space 
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

# %%
import stable_baselines3
from stable_baselines3 import A2C

# %%
model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1_000);
vec_env=model.get_env()

# %%
# Observation and action space 
obs_space = vec_env.observation_space
action_space = vec_env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))


# %%
obs = vec_env.reset()
num_episodes = 10
frames = []
for i in range(num_episodes):
    for j in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        if i==num_episodes-1:
            frames.append(vec_env.render())
        # vec_env.render()
        # VecEnv resets automatically
        if done:
            obs = vec_env.reset()
            print('done '+str(i))
            break

# %%
os.makedirs(imgs_dir, exist_ok=True)

filename = imgs_dir+'FIRST'+'.webm'
ffmpegio.video.write(filename, 10, np.array(frames[0::3]),overwrite=True,show_log=True)
display.HTML("""<video alt="test" controls><source src="""+filename+""" type="video/webm"></video>""")

# %%
model.learn(total_timesteps=1_000);

# %%
vec_env=model.get_env()

# %%
obs = vec_env.reset()
num_episodes = 10
frames = []
for i in range(num_episodes):
    for j in range(100000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        if i==num_episodes-1:
            frames.append(vec_env.render())
        # vec_env.render()
        # VecEnv resets automatically
        if done:
            obs = vec_env.reset()
            print('done '+str(i))
            break

# %%
os.makedirs(imgs_dir, exist_ok=True)

filename = imgs_dir+'SECOND'+'.webm'
ffmpegio.video.write(filename, 10, np.array(frames[0::3]),overwrite=True,show_log=True)
display.HTML("""<video alt="test" controls><source src="""+filename+""" type="video/webm"></video>""")

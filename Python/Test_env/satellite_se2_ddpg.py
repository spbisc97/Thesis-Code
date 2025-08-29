# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: default-RL
#     language: python
#     name: python3
# ---

# %%
import safegym

# %%
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
import numpy as np
import sys
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from gymnasium.wrappers.time_limit import TimeLimit


from safegym.envs import Satellite_SE2
from gymnasium.envs.registration import register
register(
    id='Satellite-SE2-v0',
    entry_point='Satellite_SE2:Satellite_SE2',
    max_episode_steps=500,
    reward_threshold=475.0,
)
import os
starting_state=np.array([0,0,0,0,0,0,0,0])
starting_noise=np.array([60,60,0.1,0.1,0.1,0,0,0])

import multiprocessing

nproc=multiprocessing.cpu_count()
print(nproc)

env = gym.make('Satellite-SE2',starting_state=starting_state,starting_noise=starting_noise)
#env = DummyVecEnv([lambda:env])
#env=Timelimit(env,max_episode_steps=500)

#env = make_vec_env(env_name, n_envs=n_envs)

# The noise objects for DDPG
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


model=DDPG("MlpPolicy",env,action_noise=action_noise,verbose=1)

# %%
model.learn(1000,progress_bar=True, log_interval=1,reset_num_timesteps=False)
model.save("Satellte-DDPG")


# %%
# # !pip install -q IPython
# %cd /run/media/simone/Shared/Documenti/Magistrale/Tesi/Code
model.load("Satellte-DDPG")
# from IPython.core.display import Image
import numpy as np
starting_state=np.array([0,0,0,0,0,0,0,0])
starting_noise=np.array([60,60,0.1,0.1,0.1,0,0,0])
import matplotlib
from moviepy.editor import ImageSequenceClip


# # %matplotlib notebook
# matplotlib.use("nbagg")
del env
env = gym.make("Satellite-SE2",render_mode="rgb_array",starting_state=starting_state,starting_noise=starting_noise)
term=False
obs,info=env.reset()
frames=[env.render()]
env.reset()
for _ in range(10000):
  action,t=model.predict(obs, deterministic=True)
  obs, reward, term, trunc, info = env.step( action)
  if _ % 50 == 0:
    frames.append(env.render())
env.close()

clip = ImageSequenceClip((frames), fps=200)
clip.write_gif('test.gif', fps=200)
# Image('test.gif')

# %%
# %timeit env.step(action)

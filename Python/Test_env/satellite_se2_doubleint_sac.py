# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from safegym.envs import Satellite_SE2
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
import os
import numpy as np
from matplotlib import pyplot as plt
import time


# %%
env=gym.make("Satellite-SE2",doubleintegrator=True,underactuated=False,unconstrained=True)
env=TimeLimit(env,5000)
model=SAC("MlpPolicy",env,verbose=1,learning_rate=0.0001,batch_size=256,learning_starts=1000,buffer_size=1000000,tau=0.005,gamma=0.999,train_freq=(1,"episode"),gradient_steps=100,ent_coef="auto")
model.learn(total_timesteps=1_000_000,log_interval=1,progress_bar=True)

# %%
from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model,env,n_eval_episodes=1000,render=False,return_episode_rewards=False)

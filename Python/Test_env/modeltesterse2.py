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
from stable_baselines3 import DDPG,TD3,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from safegym.envs import Satellite_rot
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
import os
import numpy as np
from matplotlib import pyplot as plt
import time


# %%
def classic_test_env():
    import PIL.Image as Image

    from safegym.envs.Satellite_rot import Chaser
    env = gym.make(
        "Satellite-rot-v0",
        render_mode="rgb_array",
        control="PID",
        matplotlib_backend=None,
    )
    env=TimeLimit(env,max_episode_steps=500)
    print("env checked")
    term = False

    obs, info = env.reset()
    while True:
        action = Chaser.quaternion_err_rate(
            obs[0:4], info["qd"], w=obs[4:8]
        )
        action = np.clip(action, -Chaser.Tmax, Chaser.Tmax)
        # action = env.action_space.sample()
        # action = np.array([0, 1, 0])
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            print(term, trunc)
            x = env.render()
            img = Image.fromarray(x)
            img.show()
            img.save(f"classic.jpg")
            break
    env.close()



# %%
def test__model_env(model):
    import PIL.Image as Image

    env = gym.make(
        "Satellite-rot-v0",
        render_mode="rgb_array",
    )
    env=TimeLimit(env,max_episode_steps=500)
    term = False
    obs, info = env.reset()
    while True:
        action,info = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            print(term, trunc)
            x = env.render()
            img = Image.fromarray(x)
            img.show()
            img.save(f"ppo.jpg")
            break
    env.close()



# %%
def env_maker(render_mode=None):
    env = gym.make(
        "Satellite-SE2-v0",
        render_mode=render_mode,
        step=np.float32(0.1),
        unconstrained=True,
        underactuated=True,
    )

    env = TimeLimit(env, max_episode_steps=20_000)
    env = Monitor(env)

    return env


# %%
env_name = "Satellite-rot-v0"
Algo_name = "PPO"
date = "05_15_16_36"
last_model = "52"


models_dir = "/run/media/simone/Shared/Documenti/Magistrale/Tesi/Code/Python/Test_env/savings/" 
models_dir+= f"{env_name}/{Algo_name}/{date}"#/models"
model_position=f"{models_dir}/{Algo_name}_{last_model}"
# print(str(model_position))

# model_position= "/run/media/simone/Shared/Documenti/Magistrale/Tesi/Code/Python/Test_env/savings/Satellite-SE2-v0/TD3/10_08_20_15/models/TD3_1"
# print(model_position)
model=PPO.load(str(model_position))
test__model_env(model)

# %%
classic_test_env()

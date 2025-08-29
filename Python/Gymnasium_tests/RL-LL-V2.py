# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import os
import gymnasium as gym
import matplotlib.pyplot as plt 
# # %matplotlib inline
# os.environ["SDL_VIDEODRIVER"] = "dummy"
from IPython import display

# %%
import ffmpegio
import numpy as np

# %%
from stable_baselines3 import A2C
Algo=A2C
Algo.name = "A2C"

# %%
env = gym.make(
    "LunarLander-v2",
    continuous  = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    render_mode='rgb_array')

observation, info = env.reset(seed=42)
env_name="LunarLander-v2"

# %%
models_dir=f"models/{env_name}/{Algo.name}"
logdir = f"logs/{env_name}/{Algo.name}"
imgs_dir = f"imgs/{env_name}/{Algo.name}"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)

# %%
# %load_ext tensorboard
# %tensorboard --logdir {logdir}

# %% [markdown]
# ## Print Agent Information

# %%
print("Observation Space: ", format(env.observation_space))
print("Sample Observation", format(env.observation_space.sample()))


# %%

print("Action Space       ", format(env.action_space))
print("Action Space Sample ", format(env.action_space.sample()))


# %%
model = Algo("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
vec_env = model.get_env()

# %%
print("Observation Space: ", format(vec_env.observation_space))
print("Sample Observation", format(vec_env.observation_space.sample()))

# %%
print("Action Space       ", format(vec_env.action_space))
print("Action Space Sample ", format(vec_env.action_space.sample()))

# %% [markdown]
# Save Models

# %%
last_run = 1

# %%
timesteps = 40_000
for i in range(last_run ,last_run + 5):
    model.learn(total_timesteps=timesteps,reset_num_timesteps=False,tb_log_name="run_"+str(format(i,'04d')))
    model.save(f"{models_dir}/{Algo.name}_{format(i,'04d')}")
    choosen_model_name=f"{Algo.name}_{format(i,'04d')}"

# %% [markdown]
# ## Show Whats Learned

# %%
# Number of steps you run the agent for 
num_episodes = 2

# %%
# we can now remake the env with human mode so we can render it
env = gym.make(
    "LunarLander-v2",
    continuous  = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    render_mode='rgb_array')
# we could also change some parameters of the environment to check robustness of the agent

# %%
# we can now use the one of the models that we have saved to run the agent
choosen_model = Algo.load(f"{models_dir}/{choosen_model_name}",env=env)

# %%
for ep in range(num_episodes):
    obs,info=env.reset()
    term=False
    frames=[]
    while not term or trunc:
        action, _state = choosen_model.predict(obs, deterministic=True)
        obs, reward, term,trunc, info = env.step(action)
        if ep==num_episodes-1:
            frames+=[env.render()]
        if term or trunc:
            break

# %%
filename = f"./{imgs_dir}/{choosen_model_name}.webm"
ffmpegio.video.write(filename, 10, np.array(frames[0::3]),overwrite=True,show_log=True)
display.HTML(f"""<video alt="test" controls><source src="""+filename+""" type="video/webm"></video>""")

# %%
env.close()

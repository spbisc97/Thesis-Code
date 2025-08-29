# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: SafeRL
#     language: python
#     name: python3
# ---

# %%
import gymnasium as gym

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
import stable_baselines3
from stable_baselines3 import A2C

print(stable_baselines3.__version__)

# %%
env = gym.make("CartPole-v1", render_mode="rgb_array")
env_name = "CartPole-v1"
Algo_name = "A2C"
imgs_dir = f"imgs/{env_name}/{Algo_name}"

# %%
model = A2C("MlpPolicy", env, verbose=0)

# %%
model.learn(total_timesteps=10_000)

# %%
vec_env=model.get_env()

# %%

obs = vec_env.reset()
num_episodes = 10

# %%
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

filename = imgs_dir +'FIRST'+'.webm'
options={'c:v':'libvpx-vp9','crf':'30','deadline':'realtime'}
ffmpegio.video.write(filename, 30, np.array(frames[0::1]),overwrite=True,show_log=True,**options)
from base64 import b64encode
mp4 = open(filename,'rb').read()
data_url = "data:video/webm;base64," + b64encode(mp4).decode()
#display.HTML(f"""<video alt="test" controls><source src="""+filename+""" type="video/mp4"></video>""")
display.HTML("""
<video width=400 controls>
      <source src="%s" type="video/webm">
</video>
""" % data_url)

# %%
model.learn(total_timesteps=10_000)
vec_env=model.get_env()

obs = vec_env.reset()
num_episodes = 10

# %%
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

filename = imgs_dir+'SECOND'+'.mp4'
ffmpegio.video.write(filename, 30, np.array(frames[0::1]),overwrite=True,show_log=True)
#display.HTML(f"""<video alt="test" controls><source src="""+filename+""" type="video/ogg"></video>""")
from base64 import b64encode
mp4 = open(filename,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
#display.HTML(f"""<video alt="test" controls><source src="""+filename+""" type="video/ogg"></video>""")
display.HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

# %%
vec_env.close()

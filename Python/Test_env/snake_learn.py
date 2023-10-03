import numpy as np
import os

from PyQt5.QtCore import QLibraryInfo  # others imports

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2

img = np.zeros((500, 500, 3), dtype="uint8")
cv2.imshow("Test", img)
cv2.waitKey(10)
cv2.destroyAllWindows()

# prequel
# needed for cv2.imshow to work


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Python.SafeGym.safegym.envs import SnakeEnv
import gymnasium as gym
import numpy as np
import os
import time
import ffmpegio


file_path = os.path.dirname(__file__)
os.chdir(file_path)

env_name = "Snake-v0"
# env = gym.make(env_name)

Algo = PPO
Algo.name = "PPO"

use_last_model = False

if use_last_model:
    date = input("Insert date: ")
    last_model = int(input("Insert model number: "))

if not use_last_model:
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    last_model = 0

print({"date": date, "last_model": last_model})
time.sleep(5)

top_dir = "savings/"
models_dir = top_dir + f"{env_name}/{Algo.name}/{date}/models/"
logdir = top_dir + f"{env_name}/{Algo.name}/{date}/logs/"
imgs_dir = top_dir + f"{env_name}/{Algo.name}/{date}/imgs/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)

env = make_vec_env(env_name, n_envs=4)


TIMESTEPS = 200_000
last_model = 155
if last_model > 0:
    model = Algo.load(
        f"{models_dir}/{Algo.name}_{last_model}",
        env=env,
        verbose=1,
        tensorboard_log=logdir,
    )
else:
    model = Algo("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

episodes = 0
for i in range(last_model + 1, last_model + episodes + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"run_{i}",
    )
    model.save(f"{models_dir}/{Algo.name}_{i}")
    last_model = i


env = gym.make("Snake-v0", render_mode="human")

episodes = 3


for episode in range(episodes):
    obs, info = env.reset()
    frames = []
    frames += [env.render()]
    term = False
    while not term:
        action, _states = model.predict(obs)
        obs, reward, term, trunc, info = env.step(action)
        if episode == episodes - 1:
            frames += [env.render()]

        if term or trunc:
            term = False
            break

print(len(frames))
if env.render_mode == "rgb_array":
    filename = f"./{imgs_dir}/{Algo.name}_{last_model}.mp4"
    ffmpegio.video.write(
        filename, 10, np.array(frames[0::3]), overwrite=True, show_log=True
    )

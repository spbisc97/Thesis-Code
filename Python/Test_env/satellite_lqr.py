import matplotlib.pyplot as plt

plt.legend()
plt.show(block=False)
plt.close("all")
import numpy as np
import time
from Envs.Satellite import Satellite_base
import gymnasium as gym

# from stable_baselines3 import DDPG
# from stable_baselines3.common.env_util import make_vec_env

# import os
# import send2trash
# import time
# import ffmpegio


# file_path = os.path.dirname(__file__)
# os.chdir(file_path)

env_name = "Satellite-v0"
# env = gym.make(env_name)

# Algo = "LQR"
Algo_name = "LQR"


# models_dir = f"models/{env_name}/{Algo_name}"
# logdir = f"logs/{env_name}/{Algo_name}"
# imgs_dir = f"imgs/{env_name}/{Algo_name}"
#
# os.makedirs(models_dir, exist_ok=True)
# os.makedirs(logdir, exist_ok=True)
# os.makedirs(imgs_dir, exist_ok=True)

# env = make_vec_env(env_name, n_envs=1)
env = gym.make(env_name, control="LQR")

K = np.array(
    [
        [
            0.000110841893844189,
            -9.42863182858601e-06,
            -1.3683699332241e-20,
            0.0311588547878855,
            0.0416078206376348,
            -2.60540076521516e-17,
        ],
        [
            2.8691835235862e-05,
            -6.08300962818426e-07,
            9.17052086490964e-22,
            0.00138692735458782,
            0.0124467361240575,
            -1.15521460101992e-18,
        ],
        [
            -2.82912468323803e-19,
            1.93581086037258e-20,
            8.3823164356623e-06,
            1.31190777529513e-17,
            -1.35991280487585e-16,
            0.0258956622787094,
        ],
    ]
)


episodes = 1
term = False
for episode in range(1, episodes + 1):
    obs, info = env.reset()
    rewards = [0]
    rewards_sum=[0]
    counter = [0]
    obss = [obs[0:6]]
    actions = [env.action_space.sample() * 0]

    while not term:
        action = -np.matmul(K, obs[0:6])
        # print(action)
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        rewards_sum.append(rewards_sum[-1] + reward)
        counter.append(counter[-1] + 1)
        obss.append(obs[0:6])
        actions.append(action)
        # print(obs)
        # print(np.linalg.norm(obs[0:3]))
        # time.sleep(0.02)
        if np.linalg.norm(obs[0:6]) < 0.0001 or counter[-1] > 50000:
            term = True

        if term or trunc:
            term = False
            break


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.plot(counter, rewards, label="LQR")

ax2.plot(counter, obss)
ax2.legend(["x", "y", "z", "vx", "vy", "vz"])

ax3.plot(counter, actions)
ax3.legend(["ux", "uy", "uz"])

ax4.plot(counter, rewards_sum)
ax4.legend(["LQR"])
plt.show(block=True)


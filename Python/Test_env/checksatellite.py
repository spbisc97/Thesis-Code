import numpy as np
import os

from Envs.Satellite import Satellite_base
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
import time

env = gym.make("Satellite-v0")
check_env(env)

# It will check your custom environment and output additional warnings if needed


episodes = 1
term = False
for episode in range(1, episodes+1):
    obs, info = env.reset()
    K = np.array([[3.10446912836289e-08,  -2.73186895986547e-11,    -9.88240420260271e-20,  2.69229716706869e-06,     1.31826306669043e-05, 9.67820398309341e-17],
                  [5.86774799334853e-07, -2.58150711097142e-10,   -8.02539314140983e-21,
                      4.39421022232945e-07,   0.000249254005185124,    -5.73167583337792e-18],
                   [-2.95196688268913e-18,  -1.17680560270348e-23,     9.01309369777884e-14,        9.68285856369672e-16,  -1.67494659966785e-15,   2.68563510472645e-06]])
    counter = 0
    while not term:
        action = np.matmul(K,obs[0:6])
        #action=[0,0,0]
        print("new_action")
        print(action)
        obs, reward, term, trunc, info = env.step(action)
        print(reward)
        print(obs[0:6])
        counter+=1
        
        time.sleep(0.02)

        if term or trunc:
            term = False
            break

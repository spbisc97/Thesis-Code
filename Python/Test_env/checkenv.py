import numpy as np
import os

from PyQt5.QtCore import QLibraryInfo # others imports
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

import cv2
img = np.zeros((500,500,3),dtype='uint8')
cv2.imshow('Test',img);
cv2.waitKey(10);
cv2.destroyAllWindows();

from Snake.Snake_v0 import SnakeEnv
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
import time

env = gym.make("Snake-v0",render_mode='human')
check_env(env)

# It will check your custom environment and output additional warnings if needed


episodes = 0
term=False
for episode in range(1, episodes+1):
    obs,info=env.reset()
    while not term:
        env.render()
        action_button=input("Press Enter to continue...")
        switcher = {
             'a':0,
             'd':1,
             'w':3,
             's':2,
        }
        action=switcher.get(action_button,1)
        print(action)
        obs,reward,term,trunc,info= env.step(action)
        print(reward)

        if term:
            term=False
            break

episodes = 1
term=False
for episode in range(1, episodes+1):
    obs,info=env.reset()
    counter=0
    switcher = {
             "1":1,
             "2":2,
             "3":0,
             "4":3,
        }
    while not term:
        counter+=1

        env.render()
        key=int(1+np.floor(counter/4)%4)
        action=switcher.get(str(key),1)
        print(str(key)+"  "+str(action))
        time.sleep(0.001)
        obs,reward,term,trunc,info= env.step(action)
        print(reward)

        if term:
            term=False
            break
import numpy as np
import os


from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from Envs.Satellite import Satellite_base
import gymnasium as gym
import numpy as np
import os
import send2trash
import time
import ffmpegio


file_path=os.path.dirname(__file__)
os.chdir(file_path)

env_name = "Satellite-v0"
#env = gym.make(env_name)

Algo = "A2C"
Algo.name="A2C"



models_dir=f"models/{env_name}/{Algo.name}"
logdir = f"logs/{env_name}/{Algo.name}"
imgs_dir = f"imgs/{env_name}/{Algo.name}"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)

env = make_vec_env(env_name, n_envs=1)
#env=gym.make(env_name)



TIMESTEPS = 100_000
last_model = 0
if last_model>0:
    model=Algo.load(f"{models_dir}/{Algo.name}_{last_model}",env=env, verbose=1, tensorboard_log=logdir)
else:
    input("Press Enter to delete logs and models")
    send2trash.send2trash(f"{logdir}/")
    send2trash.send2trash(f"{models_dir}/")
    model=Algo('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    
episodes=100
for i in range(last_model+1,last_model+episodes+1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,tb_log_name=f"run_{i}")
    model.save(f"{models_dir}/{Algo.name}_{i}")
    last_model=i
    
env=gym.make(env_name)

episodes = 1
term=False
for episode in range(1, episodes+1):
    obs,info=env.reset()
    
    while not term:
        action,_states =model.predict(obs)
        obs,reward,term,trunc,info= env.step(action)
        if np.linalg.norm(obs[0:3])<10:
            print(action)
            print(reward)
            print(obs)

        if term or trunc:
            term=False
            break
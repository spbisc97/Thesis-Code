import matplotlib.pyplot as plt

plt.legend()
plt.show(block=False)
plt.close("all")


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Envs.Satellite import Satellite_base
import gymnasium as gym
import numpy as np
import os
import send2trash
import time
import ffmpegio


file_path = os.path.dirname(__file__)
os.chdir(file_path)

env_name = "Satellite-discrete-v0"
# env = gym.make(env_name)

Algo = PPO
Algo.name = "PPO"


models_dir = f"models/{env_name}/{Algo.name}"
logdir = f"logs/{env_name}/{Algo.name}"
imgs_dir = f"imgs/{env_name}/{Algo.name}"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)


def run_episode(model, env_name, model_name="PPO", model_num=0, model_timesteps=0):
    term = False
    env=gym.make(env_name)
    obs, info = env.reset()
    rewards = [0]
    counter = [0]
    obss = [obs[0:6]]
    actions = [env.action_space.sample() * 0]
    while not term:
        action, _states = model.predict(obs)
        obs, reward, term, trunc, info = env.step(action)

        rewards.append(reward)
        counter.append(counter[-1] + 1)
        obss.append(obs[0:6])
        actions.append(action)

        if term or trunc:
            term = False
            break
    save_plot(
        counter,
        rewards,
        obss,
        actions,
        name=model_name,
        model_num=model_num,
        model_timesteps=model_timesteps,
    )


def save_plot(
    counter, rewards, obss, actions, name="PPO", model_num=0, model_timesteps=0
):
    rewards_sum = np.zeros_like(rewards)
    for i in range(len(rewards)):
        rewards_sum[i] = np.sum(rewards[:i])
    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2)

    ax1.plot(counter, rewards, label="PPO")

    ax2.plot(counter, obss)
    ax2.legend(["x", "y", "z", "vx", "vy", "vz"])

    ax3.plot(counter, actions)
    ax3.legend(["ux", "uy", "uz"])
    
    ax4.plot(counter, rewards_sum)
    ax4.legend(["PPO"])

    plt.savefig(f"{imgs_dir}/{name}_{model_num}_{model_timesteps:.1e}.png")
    plt.close("all")


env = make_vec_env(env_name, n_envs=1)
# env=gym.make(env_name)


TIMESTEPS = 50_000
last_model = 0
if last_model > 0:
    model = Algo.load(
        f"{models_dir}/{Algo.name}_{last_model}",
        env=env,
        verbose=1,
        tensorboard_log=logdir,
    )
else:
    input("Press Enter to delete logs and models")
    send2trash.send2trash(f"{logdir}/")
    send2trash.send2trash(f"{models_dir}/")
    model = Algo("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

episodes = 300
for i in range(last_model + 1, last_model + episodes + 1):
    model.learn(
        total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"run_{i}",log_interval=1
    )
    model.save(f"{models_dir}/{Algo.name}_{i}")
    last_model = i
    run_episode(model, env_name=env_name, model_name=Algo.name, model_num=last_model, model_timesteps=model.num_timesteps)




import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from Envs.Satellite_rot import Satellite_rot
import gymnasium as gym
import numpy as np
import os
import send2trash
import time
import ffmpegio


file_path = os.path.dirname(__file__)
os.chdir(file_path)

env_name = "Satellite-rot-v0"
# env = gym.make(env_name)

Algo = PPO
Algo.name = "PPO"
# ENT = 0.01
use_last_model = False

if use_last_model:
    if "MODEL_DATE" not in os.environ:
        raise ValueError("MODEL_DATE not in os.environ")
    if "MODEL_NUM" not in os.environ:
        raise ValueError("MODEL_NUM not in os.environ")
    date = os.environ["MODEL_DATE"]
    last_model = int(os.environ["MODEL_NUM"])

if not use_last_model:
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    os.environ["MODEL_DATE"] = date
    last_model = 0

print({"date": date, "last_model": last_model})
time.sleep(5)

models_dir = f"models/{env_name}/{date}/{Algo.name}"
logdir = f"logs/{env_name}/{date}/{Algo.name}"
imgs_dir = f"imgs/{env_name}/{date}/{Algo.name}"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)


def run_episode(
    model, env_name, model_name="PPO", model_num=0, model_timesteps=0, args=()
):
    term = False
    env = gym.make(env_name, render_mode="rgb_array")
    obs, info = env.reset()
    while not term:
        action, _states = model.predict(obs)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            X = env.render()
            plt.imsave(
                f"{imgs_dir}/{model_name}_{model_num}_{model_timesteps:.1e}.png", X
            )
            term = False
            break


env = make_vec_env(env_name, n_envs=int(os.cpu_count() / 2))
# env = gym.make(env_name)


TIMESTEPS = 50_000
last_model = 0
if last_model > 0:
    model = Algo.load(
        f"{models_dir}/{Algo.name}_{last_model}",
        env=env,
        verbose=1,
        # ent_coef=ENT,
        tensorboard_log=logdir,
    )
else:
    # input("Press Enter to delete logs and models")
    # send2trash.send2trash(f"{logdir}/")
    # send2trash.send2trash(f"{models_dir}/")
    model = Algo(
        "MlpPolicy",
        env=env,
        verbose=1,
        # ent_coef=ENT,
        tensorboard_log=logdir,
    )
episodes = 300
run_episode(
    model,
    env_name=env_name,
    model_name=Algo.name,
    model_num=last_model,
    model_timesteps=model.num_timesteps,
    args=(),
)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
for i in range(last_model + 1, last_model + episodes + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"run_{i}",
        log_interval=1,
    )
    model.save(f"{models_dir}/{Algo.name}_{i}")
    last_model = i
    os.environ["MODEL_NUM"] = str(last_model)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    run_episode(
        model,
        env_name=env_name,
        model_name=Algo.name,
        model_num=last_model,
        model_timesteps=model.num_timesteps,
        args=(),
    )

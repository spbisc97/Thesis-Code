import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import Python.SafeGym.safegym.envs
from Python.SafeGym.safegym.envs import MujSatEnv
import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0
import numpy as np
import os

# import send2trash
import time

# import ffmpegio


def fill_reward_file(imgs_dir: str):
    import inspect

    rewfile = open(f"{imgs_dir}/Reward.md", "w")
    # print("```{python}", file=rewfile)
    # print(inspect.getsource(MujSatEnv._reward_function), file=rewfile)
    # print("```", file=rewfile)


file_path = os.path.dirname(__file__)
os.chdir(file_path)

env_name = "Satellite-mj-v0"
# env = gym.make(env_name)

Algo = PPO
Algo.name = "PPO"
# ENT = 0.01
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

fill_reward_file(imgs_dir)


def run_episode(
    model, env_name, model_name="PPO", model_num=0, model_timesteps=0, args=()
):
    term = False
    env = gym.make(
        env_name,
        render_mode="rgb_array",
        camera_id=0,
    )
    env = RecordVideoV0(
        env,
        video_folder=imgs_dir,
        video_length=1000,
        name_prefix=f"{model_name}_{model_num}_{model_timesteps:.1e}",
    )
    obs, info = env.reset()
    while not term:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            X = env.render()
            plt.imsave(
                f"{imgs_dir}/{model_name}_{model_num}_{model_timesteps:.1e}.png",
                X,
            )
            term = False
            break
    env.close()


n_envs = int(os.cpu_count() / 2)
env = make_vec_env(env_name, n_envs=n_envs)

# env = gym.make(env_name)


TIMESTEPS = 250_000
if last_model > 0:
    model = Algo.load(
        f"{models_dir}/{Algo.name}_{last_model}",
        env=env,
        verbose=1,
        # ent_coef=ENT,
        tensorboard_log=logdir,
    )
else:
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
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=2, deterministic=True
)
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
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=2, deterministic=True
    )
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    run_episode(
        model,
        env_name=env_name,
        model_name=Algo.name,
        model_num=last_model,
        model_timesteps=model.num_timesteps,
        args=(),
    )

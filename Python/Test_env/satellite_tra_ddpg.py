import matplotlib as mpl

mpl.use("TkAgg")

import matplotlib.pyplot as plt


from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from Envs.Satellite_tra import Satellite_tra
import gymnasium as gym
import numpy as np
import os

# import code

# import send2trash
import time

# import ffmpegio
MAX_EPISODE_STEPS = 50_000


def fill_reward_file(imgs_dir: str):
    import inspect

    rewfile = open(f"{imgs_dir}/Reward.md", "w")
    print("```{python}", file=rewfile)
    print(inspect.getsource(Satellite_tra._reward_function), file=rewfile)
    print("```", file=rewfile)


file_path = os.path.dirname(__file__)
os.chdir(file_path)

env_name = "Satellite-tra-v0"
# env = gym.make(env_name)

Algo = DDPG
Algo.name = "DDPG"
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
    model, env_name, model_name="DDPG", model_num=0, model_timesteps=0, args=()
):
    term = False
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    obs, info = env.reset()

    while not term:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            X = env.render()
            plt.imsave(
                f"{imgs_dir}/{model_name}_{model_num}_{model_timesteps:.1e}.png", X
            )
            break
    env.close()


def create_env():
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env)
    return env


n_envs = int(os.cpu_count() / 2)
# env = make_vec_env(env_name, n_envs=n_envs)


env = create_env()

# env = DummyVecEnv([create_env for _ in range(n_envs)])

test_env = gym.make(env_name)
test_env = gym.wrappers.TimeLimit(test_env, max_episode_steps=MAX_EPISODE_STEPS)


n_actions = 3
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.005 * np.ones(n_actions)
)


TIMESTEPS = 50_000
if last_model > 0:
    model = Algo.load(
        f"{models_dir}/{Algo.name}_{last_model}",
        env=env,
        action_noise=action_noise,
        train_freq=(1, "episode"),  # try step training with multiple envs
        verbose=1,
        gamma=1,
        learning_starts=200,
        # ent_coef=ENT,
        tensorboard_log=logdir,
    )
else:
    model = Algo(
        "MlpPolicy",
        env=env,
        action_noise=action_noise,
        train_freq=(1, "episode"),  # try step training with multiple envs
        verbose=1,
        gamma=1,
        learning_starts=200,
        # ent_coef=ENT,
        tensorboard_log=logdir,
    )
# code.interact(local=locals())
episodes = 100
run_episode(
    model,
    env_name=env_name,
    model_name=Algo.name,
    model_num=last_model,
    model_timesteps=model.num_timesteps,
    args=(),
)
mean_reward, std_reward = evaluate_policy(
    model, test_env, n_eval_episodes=1, deterministic=True
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
        model, test_env, n_eval_episodes=2, deterministic=True
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

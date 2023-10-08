from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from safegym.envs import Satellite_SE2
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
import os
import numpy as np
from matplotlib import pyplot as plt
import time


def fill_reward_file(imgs_dir: str):
    import inspect

    rewfile = open(f"{imgs_dir}/Reward.md", "w")
    print("```{python}", file=rewfile)
    print(inspect.getsource(Satellite_SE2._reward_function), file=rewfile)
    print("```", file=rewfile)


file_path = os.path.dirname(__file__)
os.chdir(file_path)

env_name = "Satellite-SE2-v0"
# env = gym.make(env_name)

Algo = TD3
Algo_name = "TD3"
# ENT = 0.01
use_last_model = False

if use_last_model:
    date: str = input("Insert date: ")
    last_model: int = int(input("Insert model number: "))
else:
    date: str = time.strftime("%m_%d_%H_%M", time.localtime())
    last_model: int = 0

print({"date": date, "last_model": last_model})
time.sleep(5)

top_dir = "savings/"
models_dir = top_dir + f"{env_name}/{Algo_name}/{date}/models/"
logdir = top_dir + f"{env_name}/{Algo_name}/{date}/logs/"
imgs_dir = top_dir + f"{env_name}/{Algo_name}/{date}/imgs/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)

fill_reward_file(imgs_dir)


Y0 = 5
starting_state = np.array([0, Y0, 0, Y0 / 2000, 0, 0, 0, 0])
starting_noise = np.array([0.001, 0.001, 0.1, 1e-8, 1e-8, 0.001, 0, 0])


def run_episode(
    model, env, model_name="TD3", model_num=0, model_timesteps=0, **kargs
):
    term = False
    obs, info = env.reset()
    while not term:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            X = np.array(env.render(), dtype=np.uint8)
            plt.imsave(
                f"{imgs_dir}/{model_name}_{model_num}_{model_timesteps:.1e}.png",
                X,
            )
            term = False
            break
    env.close()


def env_maker(render_mode=None):
    env = gym.make(
        env_name,
        starting_state=starting_state,
        starting_noise=starting_noise,
        render_mode=render_mode,
        step=0.1,
    )

    env = TimeLimit(env, max_episode_steps=20_000)
    env = Monitor(env)

    return env


env = make_vec_env(env_maker, n_envs=2)

# env = gym.make(env_name)
n_actions = 2

params = {
    "mean": np.zeros(n_actions),
    "sigma": 0.05 * np.ones(n_actions),
    "dtype": np.float32,
}
O_params = {
    "theta": 0.05,
    "dt": 1e-2,
    "initial_noise": None,
}
# action_noise = OrnsteinUhlenbeckActionNoise(**params, **O_params)
action_noise = NormalActionNoise(**params)

params_episode = {
    "env": env_maker(render_mode="rgb_array_graph"),
    "model_name": Algo_name,
}

params_algo = {
    "policy": "MlpPolicy",
    "env": env,
    "verbose": 1,
    "learning_rate": 0.0001,  # 0.0003,
    "buffer_size": 1_000_000,  # 100_000,
    "batch_size": 256,  # 100,
    "tau": 0.005,  # 0.005,
    "gamma": 0.999,  # 0.99,
    "learning_starts": 1000,  # 100,
    "train_freq": (4000, "step"),  # (1, "episode"),
    "gradient_steps": 100,  # -1,
    "tensorboard_log": logdir,
    "action_noise": action_noise,
    "policy_delay": 50,  # 2,
}

TIMESTEPS = 200_000
if last_model > 0:
    model = Algo.load(
        f"{models_dir}/{Algo_name}_{last_model}",
        **params_algo,
        _init_setup_model=False,
    )
else:
    # input("Press Enter to delete logs and models")
    # send2trash.send2trash(f"{logdir}/")
    # send2trash.send2trash(f"{models_dir}/")
    model = Algo(
        **params_algo,
        _init_setup_model=True,
    )
epochs = 40
run_episode(
    model,
    **params_episode,
    model_num=last_model,
    model_timesteps=model.num_timesteps,
    kargs=(),
)


mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=1, deterministic=True
)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
for i in range(last_model + 1, last_model + epochs + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"run_{i}",
        log_interval=1,
        progress_bar=True,
    )
    model.save(f"{models_dir}/{Algo_name}_{i}")
    last_model = i
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=2, deterministic=True
    )
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    run_episode(
        model,
        **params_episode,
        model_num=last_model,
        model_timesteps=model.num_timesteps,
    )

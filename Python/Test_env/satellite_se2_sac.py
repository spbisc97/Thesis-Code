# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: SafeRL
#     language: python
#     name: python3
# ---

# %%
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from safegym.envs import Satellite_SE2
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import os
import numpy as np
from matplotlib import pyplot as plt
import time



import safegym
# #!tensorboard --logdir=/app/simonerotondi/thesis/savings/ & /app/simonerotondi/libs/ngrook/ngrok http --basic-auth='simone:rotondi97' 6006 --authtoken '--' & fg

# %%
env_name = "Satellite-SE2-v0"
# env = gym.make(env_name)

Algo = SAC
Algo_name = "SAC"
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



# %%
def fill_reward_file(imgs_dir: str,**kwargs):
    import inspect
    from pprint import pprint

    rewfile = open(f"{imgs_dir}/Reward.md", "w")
    print("```{python}", file=rewfile)
    print(inspect.getsource(Satellite_SE2._reward_function), file=rewfile)
    print("```", file=rewfile)
    print("```{python}", file=rewfile)
    pprint(kwargs,stream=rewfile)
    print("```", file=rewfile)
    rewfile.close()



# %%
def run_episode(
    model, env, model_name="SAC", model_num=0, model_timesteps=0, **kargs
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


# %%
y0: np.float32 = np.float32(5)  # [m]
# STARTING_STATE=
radius: np.float32 = y0  # [m],
speed_dev: np.float32 = np.float32(0)  # [m/s],
theta: np.float32 = np.float32(0)  # [rad],
theta_dot: np.float32 = np.float32(0)  # [rad/s],
phi: np.float32 = np.float32(0)  # [rad]
phi_dot: np.float32 = np.float32(0)  # [rad/s]

# Target initial state: [phi_target, phi_dot_target]
phi_target: np.float32 = np.float32(0)  # [rad]
phi_dot_target: np.float32 = np.float32(0)  # [rad/s]

STARTING_STATE = np.array(
    [radius, speed_dev, theta, theta_dot, phi, phi_dot, phi_target, phi_dot_target], dtype=np.float32
)

radius_noise: np.float32 = np.float32(0)
speed_noise_multiplier: np.float32 = np.float32(0)
theta_noise: np.float32 = np.float32(0)
theta_dot_noise: np.float32 = np.float32(0)
phi_noise: np.float32 = np.float32(0)
phi_dot_noise: np.float32 = np.float32(0)

# Target noise: [phi_target_noise, phi_dot_target_noise]
phi_target_noise: np.float32 = np.float32(0)  # [rad]
phi_dot_target_noise: np.float32 = np.float32(0)  # [rad/s]

STARTING_NOISE = np.array(
    [
        radius_noise,
        speed_noise_multiplier,
        theta_noise,
        theta_dot_noise,
        phi_noise,
        phi_dot_noise,
        phi_target_noise,
        phi_dot_target_noise,
    ],
    dtype=np.float32,
)
initial_integration_steps = np.array([0,1], dtype=np.int32)
# REWARD_WEIGHTS = distance_decrease,-distance,-action,-speed,-angle_speed
REWARD_WEIGHTS = np.array([20, 0.5, 0.5, 1, 30], dtype=np.float32)


env_params={
    # "starting_state":STARTING_STATE,
    # "starting_noise":STARTING_NOISE,
    "initial_integration_steps":initial_integraton_steps,
    "underactuated":True,
    "step":10,
    # "reward_weights":REWARD_WEIGHTS,

}


# %%
def env_maker(render_mode=None):
    env = gym.make(
        env_name,
        render_mode=render_mode,
        **env_params
    )

    env = TimeLimit(env, max_episode_steps=60_000)
    env = Monitor(env)

    return env


# %%

env = make_vec_env(env_maker, n_envs=1)

# env = gym.make(env_name)
n_actions = 2

params = {
    "mean": np.zeros(n_actions),
    "sigma": np.array([1e-3, 1e-5], dtype=np.float32),  # np.ones(n_actions
    "dtype": np.float32,
}
O_params = {
    "theta": 0.2,
    "dt": 1e-2,
    "initial_noise": None,
}
#action_noise = OrnsteinUhlenbeckActionNoise(**params, **O_params)
action_noise = NormalActionNoise(**params)

params_episode = {
    "env": env_maker(render_mode="rgb_array_graph"),
    "model_name": Algo_name,
}

params_common_algo = {
    "policy": "MlpPolicy",
    "env": env,
    "verbose": 1,
    "tensorboard_log": logdir,
    "stats_window_size": 30,
}
params_algo = {
    **params_common_algo,
    'learning_rate':0.0003, 
    'buffer_size':1000000, 
    'learning_starts':100, 
    'batch_size':256, 
    'tau':0.005, 
    'gamma':0.999, #0.99
    'train_freq':1, 
    'gradient_steps':1, 
    'action_noise':None, 
    'replay_buffer_class':None, 
    'replay_buffer_kwargs':None, 
    'optimize_memory_usage':False,
    #'ent_coef'='auto', 
    'target_update_interval':1, 
    'target_entropy':'auto', 
    'use_sde':False,
    'sde_sample_freq':-1,
    'use_sde_at_warmup':False
}

TIMESTEPS = 5_000
params_learn = {
    "total_timesteps": TIMESTEPS,
    "reset_num_timesteps": False,
    "log_interval": 2,
    "progress_bar": False,
    #"callback": EpisodeEndCallback(),
}

# %%
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
    fill_reward_file(imgs_dir,params_algo=params_algo,params_learn=params_learn,params=params,O_params=O_params,env_params=env_params)


# %%
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



# %%
epochs=10
for i in range(last_model + 1, last_model + epochs + 1):
    model.learn(
        **params_learn,
        tb_log_name=f"run_{i}",
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

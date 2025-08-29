# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: default-RL
#     language: python
#     name: python3
# ---

# %%
import gymnasium as gym
env = gym.make('Humanoid-v4',render_mode='human')


# %%
num_episodes=10
for ep in range(num_episodes):
    obs,info=env.reset()
    term=False
    frames=[]
    while not term or trunc:
        action = env.action_space.sample()                                                                                                                                                                                                                                                                                                                                                  
        obs, reward, term,trunc, info = env.step(action)
        if term or trunc:           
            break

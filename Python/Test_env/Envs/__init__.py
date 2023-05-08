from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="Envs.Snake_v0:SnakeEnv",
)
register(
    id="Satellite-v0",
    entry_point="Envs.Satellite:Satellite_base",
    max_episode_steps=20000,
    reward_threshold=25000.0,
)
register(
    id="Satellite-discrete-v0",
    entry_point="Envs.Satellite:Satellite_base",
    max_episode_steps=15000,
    reward_threshold=25000.0,
    kwargs={"action_space": "discrete"},
)

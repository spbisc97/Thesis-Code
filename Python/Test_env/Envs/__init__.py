from gymnasium.envs.registration import register

register(
    id='Snake-v0',
    entry_point='Envs.Snake_v0:SnakeEnv',
)
register(
    id='Satellite-v0',
    entry_point='Envs.Satellite:Satellite_base',
)
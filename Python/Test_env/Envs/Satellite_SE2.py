import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Optional
from gymnasium.envs.classic_control.rendering import RenderFrame



class Satellite_SE2(gym.Env):
    metadata={'render.modes': ['human', 'rgb_array',None]}
    def __init__(self,
                 render_mode: Optional[str]=None,
                 
                 
                 
                 ):
        super(Satellite_SE2, self).__init__()
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.state = None
        self.viewer = None
        self.steps_beyond_done = None
        self.seed()
        self.reset()
        
        
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
         super().reset(seed=seed, options=options)
         return
     
     
     
    def step():
        return
    
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()
    
    
    def close(self) -> None:
        super().close()
        return
    
    
    def seed(self, seed: int | None = None) -> list[int]:
        return super().seed(seed=seed)
    
    
    
    
    
    INERTIA=1
    INERTIA_INV=1
    MASS=1
    
    class Chaser:
        def __init__(self):
            self.x = 0
            self.y = 0
            self.theta = 0
            self.x_dot = 0
            self.y_dot = 0
            self.theta_dot = 0
            
    
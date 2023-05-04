
import numpy as np
import gymnasium as gym

class SatelliteEnv(gym.Env):
    def __init__(self,current_position,current_velocity):
        self.target_position = np.zeros(3)
        self.target_velocity = np.zeros(3)
        self.current_position = current_position #later on this will be random
        self.current_velocity = current_velocity #later on this will be random
        self.step_count = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

    def step(self, action):
        # Update the satellite's position and velocity based on the action
        # action is the thruster firing in x,y,z direction

        # Calculate the reward based on the distance from the target
        reward = -np.linalg.norm(self.current_position - self.target_position)

        self.step_count += 1

        # Return the observation, reward, done flag, and info dictionary
        observation = np.concatenate((self.current_position, self.current_velocity))
        terminated  = (self.step_count >= 1000)
        info = {}

        return observation, reward, terminated, False ,info

    def reset(self):
        # Reset the satellite's position and velocity to their initial values
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.step_count = 0

        # Return the initial observation
        observation = np.concatenate((self.current_position, self.current_velocity))
        return observation

    def render(self, mode='human'):
        # Visualize the current state of the environment
        pass

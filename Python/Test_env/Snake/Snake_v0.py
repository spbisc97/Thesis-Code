import gymnasium as gym
from gymnasium import spaces
import os
from PyQt5.QtCore import QLibraryInfo  # others imports

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

SNAKE_LEN_GOAL = 30
from collections import deque
import numpy as np
import random
import cv2
import time


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if (
        snake_head[0] >= 500
        or snake_head[0] < 0
        or snake_head[1] >= 500
        or snake_head[1] < 0
    ):
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        # define action and observation space being gymnaium.spaces

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-500, high=500, shape=(5 + SNAKE_LEN_GOAL,), dtype=np.int64
        )
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, action):
        reward=0
        terminated = False
        self.prev_actions.append(action)

        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down
        # Change the head position based on the button direction
        if action == 1:
            self.snake_head[0] += 10
        elif action == 0:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            reward += 1000
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score
            )
            self.snake_position.insert(0, list(self.snake_head))
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if (
            collision_with_boundaries(self.snake_head) == 1
            or collision_with_self(self.snake_position) == 1
        ):
            terminated = True
        # reward proposal
        euclidian_distance=np.sqrt((self.snake_head[0]-self.apple_position[0])**2+(self.snake_head[1]-self.apple_position[1])**2)
        reward += -euclidian_distance/500 if not terminated else -1000

        # observation proposal
        # head_x, head_y, apple_delta_x, apple_delta_y, snake_length, previous_moves
        observation = self._get_obs()
        info={}
        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )  # observation, reward, terminated,truncated, info

    def reset(self):
        terminated = False

        # self.img = np.zeros((500,500,3),dtype='uint8')
        ## Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]
        self.score = 0
        self.action = 1
        reward = 0
        self.snake_head = [250, 250]
        ...
        # observation proposal
        # head_x, head_y, apple_delta_x, apple_delta_y, snake_length, previous_moves
        # head_x = self.snake_head[0]
        # head_y = self.snake_head[1]
        # apple_delta_x = head_x -self.apple_position[0]
        # apple_delta_y =  head_y -self.apple_position[1]
        # snake_length = len(self.snake_position)
        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = self._get_obs()
        info={}

        return observation, info  # observation, info

    ## not needed
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            cv2.imshow("Snake", self._render_frame())
            cv2.waitKey(1)
            time.sleep(0.01)
            return  
        
    def _render_frame(self):
        img = np.zeros((500, 500, 3), dtype="uint8")
        # Display Apple
        cv2.rectangle(
            img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            3,
        )
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(
                img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 0),
                3,
            )
        return img

    def close(self):
        # if it is necessary to shut down the environment properly
        if self.render_mode == "human":
            cv2.destroyWindow("Snake")
        pass

    def _get_obs(self):
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]
        apple_delta_x = head_x - self.apple_position[0]
        apple_delta_y = head_y - self.apple_position[1]
        snake_length = len(self.snake_position)

        observation = np.array(
            [head_x, head_y, apple_delta_x, apple_delta_y, snake_length]
            + list(self.prev_actions)
        )
        return observation

"""
    A gym environment for simulating the motion of a satellite in 2D space.

    The Chaser is modeled as a rigid body with three degrees of freedom, and is controlled by action.
    The goal of the chaser is to approach and dock with a target spacecraft, while avoiding collisions and minimizing fuel usage.

    The observation of the system is represented by a 10-dimensional vector, consisting of:
    - the position of the chaser in R2
    - the direction vector of the chaser in R2
    - the orientation vector of the target in R2
    - the velocity of the chaser in R2
    - the angular velocity of the chaser in R1
    - the angular velocity of the target in R1

    The action space is either 2-dimensional or 3-dimensional, depending on whether the environment is underactuated or not.
    The action represents the thrust vector applied by the chaser spacecraft.

    The reward function is a combination of the negative logarithm of the distance between the chaser and target spacecrafts,
    and the squared magnitude of the thrust vector.
    """

# rest of the code...


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import (
    Any,
    Optional,
    SupportsFloat,
)
import matplotlib.pyplot as plt

INERTIA = 4.17e-2  # [kg*m^2]
INERTIA_INV = 23.9808  # 1/INERTIA
MASS = 30 + 10  # [kg]
MU = 3.986004418 * 10**14  # [m^3/s^2]
RT = 6.6 * 1e6  # [m]
NU = np.sqrt(MU / RT**3)
FMAX = 1.05e-3  # [N]
TMAX = 6e-3  # [Nm]
FTMAX = 6e-3  # just to clip with the same value for
STEP = 0.05  # [s]


class Satellite_SE2(gym.Env):
    """
    A gym environment for simulating the motion of a satellite in 2D space.

    The Chaser is modeled as a rigid body with three degrees of freedom, and is controlled by action.
    The goal of the chaser is to approach and dock with a target spacecraft, while avoiding collisions and minimizing fuel usage.

    The observation of the system is represented by a 10-dimensional vector, consisting of:
    - the position of the chaser in R2
    - the direction vector of the chaser in R2
    - the orientation vector of the target in R2
    - the velocity of the chaser in R2
    - the angular velocity of the chaser in R1
    - the angular velocity of the target in R1

    The action space is either 2-dimensional or 3-dimensional, depending on whether the environment is underactuated or not.
    The action represents the thrust vector applied by the chaser spacecraft.

    The reward function is a combination of the negative logarithm of the distance between the chaser and target spacecrafts,
    and the squared magnitude of the thrust vector.
    """

    metadata = ({"render.modes": ["human", "rgb_array", None]},)
    vrot_max = 1
    vtrans_max = 10
    vrot_min = -1
    vtrans_min = -10

    def __init__(
        self,
        render_mode: Optional[str] = None,
        underactuated: Optional[bool] = True,
        starting_state: Optional[np.ndarray] = np.zeros(
            (8,), dtype=np.float32
        ),
        starting_noise: Optional[np.ndarray] = np.array(
            [10, 10, np.pi / 2, 1e-5, 1e-5, 1e-4, 0, 0]
        ),
        unit_action_space: Optional[bool] = True,
        max_action=FTMAX,
    ):
        super(Satellite_SE2, self).__init__()
        assert isinstance(underactuated, bool)
        assert isinstance(starting_state, np.ndarray)
        assert isinstance(starting_noise, np.ndarray)
        assert isinstance(unit_action_space, bool)
        self.underactuated = underactuated
        self.unit_action_space = unit_action_space
        self.max_action = max_action
        self.starting_state = starting_state
        self.starting_noise = starting_noise

        self.build_action_space()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.steps_beyond_done = None

        self.chaser = self.Chaser(underactuated=underactuated)
        self.target = self.Target()
        self.reset()

        return

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        state = self.__generate_random_state()
        self.chaser.reset(state[0:6])
        self.target.reset(state[6:8])
        observation = self.__get_observation()
        info = {}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        self.chaser.set_control(action)
        self.chaser.step()
        terminated = self.__termination()
        truncated = False
        reward = self.__reward()
        observation = self.__get_observation()
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self) -> None:
        super().close()
        return

    def __get_observation(self) -> np.ndarray:
        w = self.chaser.get_state()
        theta = self.target.get_state()
        observation = np.zeros((10,), dtype=np.float32)
        observation[0] = w[0]
        observation[1] = w[1]
        observation[2] = np.sin(w[2])
        observation[3] = np.cos(w[2])
        observation[4] = np.sin(theta[0])
        observation[5] = np.cos(theta[0])

        observation[6] = w[3]
        observation[7] = w[4]
        observation[8] = w[5]
        observation[9] = theta[1]

        return observation

    def __generate_random_state(self):
        state = np.random.normal(self.starting_state, self.starting_noise)
        return state

    def __reward(self):
        reward = 0.5
        ch_state = self.chaser.get_state()
        ch_control = self.chaser.get_control()

        reward += (-np.log10((ch_state[0]) ** 2 + (ch_state[1]) ** 2)) - (
            ch_control[0] ** 2 + ch_control[1] ** 2
        )
        return reward

    def __action_filter(self, action):
        max_action = self.max_action
        action = action * max_action
        return action

    def __termination(self):
        if self.chaser.radius() < 1:
            return True
        else:
            return False

    def crash(self):
        if (self.chaser.radius() < 1) and (self.chaser.speed() > 1):
            return True
        else:
            return False

    def success(self):
        if self.chaser.radius() < 1 and self.chaser.speed() < 1:
            return True
        else:
            return False

    def build_action_space(self):
        if self.unit_action_space:
            max_action = 1
        else:
            max_action = self.max_action
        if self.underactuated:
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(2,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(3,), dtype=np.float32
            )

    class Chaser:
        """Chaser class for the satellite environment."""

        def __init__(self, step=STEP, state=None, underactuated=True):
            self.state = None

            if state is None:
                self.set_state()
            else:
                self.set_state(state)
            assert underactuated is True | False
            self.underactuated = underactuated

            if underactuated:
                self.control = np.zeros((2,), dtype=np.float32)
                self.control_space = 2  # avoid gym space check
            else:
                self.control = np.zeros((3,), dtype=np.float32)
                self.control_space = 3  # avoid gym space check
                # would be nice to have a check control space each time but would slow down the code
            return

        def set_state(self, state=np.zeros((6,), dtype=np.float32)):
            self.state = np.float32(state)
            return

        def set_control(self, control: np.ndarray = None):
            self.control = np.float32(control)
            return

        def get_state(self):
            return self.state

        def get_control(self):
            return self.control

        def __sat_dyn(self, t: SupportsFloat, w: np.ndarray, u: np.ndarray):
            dw = np.zeros((6,), dtype=np.float32)
            torque = np.zeros((3,), dtype=np.float32)
            if self.underactuated:
                torque = np.array(
                    [np.sin(w[2]) * u[0], np.cos(w[2]) * u[0], u[1]]
                )
            else:
                torque = u

            dw[0] = w[3]
            dw[1] = w[4]
            dw[2] = w[5]
            dw[3] = (
                (3 * (NU**2) * w[0]) + (2 * NU * w[4]) + (torque[0] / MASS)
            )
            dw[4] = (-2 * NU * w[5]) + (torque[2] / MASS)
            dw[5] = INERTIA_INV * torque[1]
            return dw

        def step(self, ts=STEP):
            t = 0
            w = self.get_state()
            u = self.get_control()
            k1 = self.__sat_dyn(t, w, u)
            # self.set_state(w + ts * (k1 ))
            k2 = self.__sat_dyn(t + 0.5 * ts, w + 0.5 * ts * k1, u)
            k3 = self.__sat_dyn(t + 0.5 * ts, w + 0.5 * ts * k2, u)
            k4 = self.__sat_dyn(t + ts, w + ts * k3, u)
            self.set_state(w + ts * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            return self.state

        def reset(self, state=np.zeros((6,), dtype=np.float32)):
            self.set_state(state)
            self.set_control()
            return self.state

        def render(self):
            pass

        def radius(self):
            return np.sqrt(self.state[0] ** 2 + self.state[1] ** 2)

        def speed(self):
            return np.sqrt(self.state[3] ** 2 + self.state[4] ** 2)

    class Target:
        # add dynamics later
        def __init__(self, step=STEP, state=np.zeros((2,), dtype=np.float32)):
            # state = [theta,theta_dot]
            self.state = state
            self.set_state(state)
            return

        def set_state(
            self, state: np.array = np.zeros((2,), dtype=np.float32)
        ):
            self.state = np.float32(state)
            return

        def get_state(self):
            return self.state

        def reset(self, state=np.zeros((2,), dtype=np.float32)):
            self.set_state(state)
            return self.state

        def __remember(self):
            pass

        # add dynamics later when speed is needed


def _test():
    from stable_baselines3.common.env_checker import check_env

    check_env(Satellite_SE2(), warn=True)
    print("env checked")


def _test2(underactuated=True):
    env = Satellite_SE2(underactuated=underactuated)
    observation, info = env.reset()
    observations = [observation]
    rewards = []
    actions = []
    for _ in range(1000):
        actions.append(env.action_space.sample())
        observation, reward, term, trunc, info = env.step(actions[-1])
        observations.append(observation)
        rewards.append(reward)
    env.close()
    plt.plot(observations)
    plt.show()
    plt.plot(actions)
    plt.show()
    plt.plot(rewards)
    plt.show()

    return


def _test3(underactuated=True):
    from stable_baselines3 import PPO

    model = PPO('MlpPolicy', 'Satellite_SE2-v0').learn(100000)

    env = gym.make('Satellite_SE2-v0', underactuated=underactuated)
    observation, info = env.reset()
    observations = [observation]
    rewards = []
    actions = []
    for i in range(1000):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, trun, term, info = env.step(action)
        observations.append(observation)
        rewards.append(reward)
        actions.append(action)
    env.close()
    plt.plot(observations)
    plt.show()
    plt.plot(actions)
    plt.show()
    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    from gymnasium.envs.registration import register

    register(
        id="Satellite_SE2-v0",
        entry_point="Satellite_SE2:Satellite_SE2",
        max_episode_steps=50000,
        reward_threshold=0.0,
    )
    _test()

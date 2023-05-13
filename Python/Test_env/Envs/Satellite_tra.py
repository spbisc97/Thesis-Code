import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt

import matplotlib.style as mplstyle

# mplstyle.use("fast")
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import jit  # or try codonpy instead
import random


# ? Maybe it's better to normalize the observation space?
# ? Maybe it's better to normalize the action space?
# ? Maybe it's better to use a discrete action space?
# ? Maybe it's better to use a continuous action space?
class Satellite_tra(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "observation_spaces": ["MlpPolicy", "MultiInputPolicy"],
        "control": ["Ml", "ModelPredictiveControl", "LQR", "PID", "Random", "Human"],
        "action_spaces": ["continuous", "discrete"],
        "matplotlib_backend": [
            "TkAgg",
            "Qt5Agg",
            "Qt4Agg",
            "MacOSX",
            "WX",
            "GTK3",
            "Agg",  # non gui backend
        ],
    }

    def __init__(
        self,
        render_mode=None,
        observation_space="MlpPolicy",
        action_space="continuous",
        control="Ml",
        matplotlib_backend="TkAgg",
    ):
        super(Satellite_tra, self).__init__()
        # define action and observation space being gymnasium.spaces
        assert control in self.metadata["control"]
        self.control = control

        assert action_space in self.metadata["action_spaces"]
        if action_space == "continuous":
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.int8)
        #   or action_space in self.metadata['action_spaces']
        #   could be  better to use normalized action space
        assert observation_space in self.metadata["observation_spaces"]
        if observation_space == "MlpPolicy":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
        if observation_space == "MultiInputPolicy":
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                    ),
                    "fuel": spaces.Box(
                        low=0,
                        high=Chaser.initial_fuel_mass,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )
        # or render_mode in self.metadata['render_modes']
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert matplotlib_backend in self.metadata["matplotlib_backend"]

        mpl.use(matplotlib_backend)
        self.render_mode = render_mode
        self.subplots = None

        self.dmax = 50000
        self.vmax = 2000
        self.prev_shaping = None

    def step(self, action):
        reward = 0
        info = {}
        truncated = False
        filtered_action = self._action_filter(action)
        self.chaser.move(filtered_action)
        terminated = self._beyond_observational_space()  # or self.chaser.fuel_mass <= 0
        observation = self._get_observation()
        reward = self._reward_function(action, terminated)
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._remember(self.chaser.state, filtered_action, reward, info)
            if self.render_mode == "human":
                self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_shaping = None
        self.subplots = None
        self.chaser = Chaser()
        state = np.zeros((6,), dtype=np.float32)
        # state[random.randint(0,2)] = random.randint(-100, 100)
        state[0:3] = (np.random.rand(1, 3) - 0.5) * 20
        state[3:] = (np.random.rand(1, 3) - 0.5) * 0.001
        self.chaser.set_state(state)
        self.prev_shaping = self._shape_reward()
        info = {}
        observation = self._get_observation()
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.statuses = np.array(self.chaser.state)
            self.actions = np.array([]).reshape(0, 3)
            self.rewards = np.array([]).reshape(0, 1)
            self.rewards_sum = np.array([]).reshape(0, 1)
            self.infos = np.array(info)
            self.times = np.array([0])
        return observation, info

    def render(self):
        if not self.render_mode or (
            self.render_mode == "human" and self.times[-1] % 20 != 0
        ):
            return
        if self.subplots == None:
            fig, ax = plt.subplots(4, 3, figsize=(10, 10))
            fig.add_gridspec(hspace=30)
            lines = np.ma.zeros_like(ax)
            legend = np.array(
                [
                    ["x", "y", "z"],
                    ["vx", "vy", "vz"],
                    ["ux", "uy", "uz"],
                    ["reward", "reward_sum", ""],
                ]
            )
            limits = np.array(
                [
                    [[-100, 100], [-100, 100], [-100, 100]],
                    [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]],
                    [[-1, 1], [-1, 1], [-1, 1]],
                    [[-2, 0.3], [-100, 1], [-1, 1]],
                ]
            )
            limits[2, :] = limits[2, :] * Chaser.Tmax
            line_width = 0.6
            (lines[0, 0],) = ax[0, 0].plot(
                self.times, self.statuses[:, 0], linewidth=line_width
            )
            (lines[0, 1],) = ax[0, 1].plot(
                self.times, self.statuses[:, 1], linewidth=line_width
            )
            (lines[0, 2],) = ax[0, 2].plot(
                self.times, self.statuses[:, 2], linewidth=line_width
            )
            (lines[1, 0],) = ax[1, 0].plot(
                self.times, self.statuses[:, 3], linewidth=line_width
            )
            (lines[1, 1],) = ax[1, 1].plot(
                self.times, self.statuses[:, 4], linewidth=line_width
            )
            (lines[1, 2],) = ax[1, 2].plot(
                self.times, self.statuses[:, 5], linewidth=line_width
            )
            (lines[2, 0],) = ax[2, 0].plot(
                self.times[:-1], self.actions[:, 0], linewidth=line_width
            )
            (lines[2, 1],) = ax[2, 1].plot(
                self.times[:-1], self.actions[:, 1], linewidth=line_width
            )
            (lines[2, 2],) = ax[2, 2].plot(
                self.times[:-1], self.actions[:, 2], linewidth=line_width
            )
            (lines[3, 0],) = ax[3, 0].plot(
                self.times[:-1], self.rewards[:], linewidth=line_width
            )
            (lines[3, 1],) = ax[3, 1].plot(
                self.times[:-1], self.rewards_sum[:], linewidth=line_width
            )
            # (lines[3, 2],) = ax[3, 2].plot(self.times, self.rewards[:, 2])

            for idx, x in np.ndenumerate(ax):
                # ax[idx[0], idx[1]].set_xlim(0, 100)
                ax[idx[0], idx[1]].set_ylim(limits[idx[0], idx[1], :])
                ax[idx[0], idx[1]].set_title(legend[idx[0], idx[1]])
                ax[idx[0], idx[1]].grid()
                if idx[0] != 3 and idx[1] != 0:
                    ax[idx[0], idx[1]].sharey(ax[idx[0], idx[1] - 1])
            plt.grid()
            if self.render_mode == "human":
                plt.ion()
                plt.show(block=False)
            else:
                plt.ioff()
            self.subplots = (fig, ax, lines)

        else:
            fig, ax, lines = self.subplots
            lines[0, 0].set_data(self.times, self.statuses[:, 0])
            lines[0, 1].set_data(self.times, self.statuses[:, 1])
            lines[0, 2].set_data(self.times, self.statuses[:, 2])
            lines[1, 0].set_data(self.times, self.statuses[:, 3])
            lines[1, 1].set_data(self.times, self.statuses[:, 4])
            lines[1, 2].set_data(self.times, self.statuses[:, 5])
            lines[2, 0].set_data(self.times[:-1], self.actions[:, 0])
            lines[2, 1].set_data(self.times[:-1], self.actions[:, 1])
            lines[2, 2].set_data(self.times[:-1], self.actions[:, 2])
            lines[3, 0].set_data(self.times[:-1], self.rewards[:])
            lines[3, 1].set_data(self.times[:-1], self.rewards_sum[:])

        # mplstyle.use("fast")

        for idx, x in np.ndenumerate(ax):
            ax[idx[0], idx[1]].relim(visible_only=True)
            ax[idx[0], idx[1]].autoscale(enable=True, axis="both", tight=False)

        blit = True
        if blit:
            for idx, x in np.ndenumerate(ax):
                if lines[idx[0], idx[1]]:
                    ax[idx[0], idx[1]].draw_artist(lines[idx[0], idx[1]])
                    # ax[idx[0], idx[1]].draw_artist(ax[idx[0], idx[1]].patch)

            # for x in ax:
            #     for y in x:
            #         fig.canvas.blit(y.bbox)
            #         #memory leaks
        else:
            fig.canvas.draw()
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig.canvas.flush_events()
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        # if self.render_mode == "human":
        # fig.show()

        return

    def _remember(self, sta, act, rew, inf={}):
        self.statuses = np.vstack((self.statuses, sta))
        self.actions = np.vstack((self.actions, act))
        self.rewards = np.vstack((self.rewards, rew))
        prev_rewards = 0 if len(self.rewards_sum) == 0 else self.rewards_sum[-1]
        self.rewards_sum = np.vstack((self.rewards_sum, prev_rewards + rew))
        self.infos = np.vstack((self.infos, inf))
        self.times = np.vstack((self.times, self.times[-1] + self.chaser.step))
        return

    def _get_observation(self):
        if isinstance(self.observation_space, spaces.Box):
            return np.concatenate(
                (
                    self.chaser.state,
                    np.asarray((self.chaser.fuel_mass,), dtype=np.float32),
                )
            )
        else:
            return {
                "state": self.chaser.state,
                "fuel": np.asarray((self.chaser.fuel_mass,), dtype=np.float32),
            }

    def _reward_function(self, action=0, terminated=False):
        reward = 0 if not terminated else -20000
        # distance_reward = prev_distance - distance
        # fuel_reward = prev_fuel - fuel
        shaping = self._shape_reward()

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Compute the difference between the current state and the desired state
        # error = np.abs(self.chaser.state[0:3])  # -(self.chaser.desired_state)

        # Define weighting factors for each state variable
        # weights = np.array([1.0, 2.0, 1.5]) * 0.0001

        # Compute the weighted sum of the error
        # weighted_error = np.dot(error, weights)

        # Position Error Term
        log_position_error_term = np.log(np.linalg.norm(self.chaser.state[:3]) + 1e-9)

        # Control Effort Term
        control_effort_term = 0.1 * np.linalg.norm(action)

        reward = reward - log_position_error_term - control_effort_term

        return float(reward)

    def _shape_reward(self):
        shaping = 0
        return shaping

    def _beyond_observational_space(self):
        if np.any(np.abs(self.chaser.state[:3]) >= self.dmax) or np.any(
            np.abs(self.chaser.state[3:]) >= self.vmax
        ):
            return True
        return False

    def _action_filter(self, action):
        if self.control == "Ml":
            action = action * Chaser.Tmax
        # action = np.clip(action, -Chaser.Tmax, Chaser.Tmax)

        if np.any(np.abs(action) > Chaser.Tmax):
            # remove in later versions
            print("action out of bounds")

        return action


class Chaser:
    mass = 30
    initial_fuel_mass = 10
    mu = 3.986004418 * 1e14
    rt = 6.6 * 1e6
    Isp = 2250
    Tmax = 1.05e-1
    g = 9.81
    dt = 1

    def __init__(self, step=None):
        # self.position = np.array([0,0,0])
        # self.velocity = np.array([0,0,0])
        self.state = np.zeros((6,), dtype=np.float32)
        # self.inertia = np.array([8.33e-2,1.08e-1,4.17e-2])
        self.fuel_mass = self.initial_fuel_mass
        self.w = np.sqrt((self.mu / (self.rt**3)), dtype=np.float32)
        self.step = self.dt if step is None else step

    def set_state(self, state):
        self.state = np.float32(state)
        return self.state

    def _Sat_Translational_Dyn(self, y, t, trust):
        dy = np.zeros(
            6,
        )
        torques = trust.copy()

        dy[0] = y[3]
        dy[1] = y[4]
        dy[2] = y[5]

        rc = ((self.rt + y[0]) ** 2 + y[1] ** 2 + y[2] ** 2) ** (1 / 2)
        eta = self.mu / (rc**3)
        total_mass = self.mass + self.fuel_mass

        dy[3] = (
            self.w**2 * y[0]
            + 2 * self.w * y[4]
            + self.mu / (self.rt**2)
            - (eta) * (self.rt + y[0])
            + torques[0] / total_mass
        )
        dy[4] = (
            -2 * self.w * y[3]
            + self.w**2 * y[1]
            - (eta) * y[1]
            + torques[1] / total_mass
        )
        dy[5] = -(eta) * y[2] + torques[2] / total_mass

        return dy

    def _Sat_Translational_Linear_Dyn(self, y, t, trust):
        dy = np.zeros(
            6,
        )
        torques = trust.copy()

        dy[0] = y[3]
        dy[1] = y[4]
        dy[2] = y[5]

        rc = ((self.rt + y[0]) ** 2 + y[1] ** 2 + y[2] ** 2) ** (1 / 2)
        eta = self.mu / (rc**3)
        total_mass = self.mass + self.fuel_mass

        dy[3] = (
            self.w**2 * y[0]
            + 2 * self.w * y[4]
            + self.mu / (self.rt**2)
            - (eta) * (self.rt + y[0])
            + torques[0] / total_mass
        )
        dy[4] = (
            -2 * self.w * y[3]
            + self.w**2 * y[1]
            - (eta) * y[1]
            + torques[1] / total_mass
        )
        dy[5] = -(eta) * y[2] + torques[2] / total_mass

        return dy

    def move(self, trust):
        trust = np.clip(trust, -Chaser.Tmax, Chaser.Tmax)
        self.state = self._rk4(
            self._Sat_Translational_Dyn, self.state, self.dt, args=(trust,)
        )
        self.fuel_mass = (
            self.fuel_mass - np.sum(np.abs(trust)) / (self.g * self.Isp) * self.dt
        )
        pass

    def distance_to_target(self, target=np.zeros((3,), dtype=np.float32)):
        return np.linalg.norm(self.state[0:3] - target)

    def velocity_norm(self):
        return np.linalg.norm(self.state[3:6])

    # @jit
    def _rk4(self, f, y0, t, args=()):
        N = int(self.step / self.dt)
        t = np.linspace(0, self.step, N + 1, dtype=np.float32)
        n = len(t)
        y = np.zeros((n, len(y0)), dtype=np.float32)
        y[0] = y0
        for i in range(n - 1):
            h = t[i + 1] - t[i]
            k1 = f(y[i], t[i], *args)
            k2 = f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3 = f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4 = f(y[i] + k3 * h, t[i] + h, *args)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y[-1]


if __name__ == "__main__":
    # Q_main()
    from stable_baselines3.common.env_checker import check_env

    check_env(Satellite_tra(), warn=True)
    print("env checked")

    env = Satellite_tra(render_mode="rgb_array", control="PID")
    K = np.array(
        [
            [
                0.000110841893844189,
                -9.42863182858601e-06,
                -1.3683699332241e-20,
                0.0311588547878855,
                0.0416078206376348,
                -2.60540076521516e-17,
            ],
            [
                2.8691835235862e-05,
                -6.08300962818426e-07,
                9.17052086490964e-22,
                0.00138692735458782,
                0.0124467361240575,
                -1.15521460101992e-18,
            ],
            [
                -2.82912468323803e-19,
                1.93581086037258e-20,
                8.3823164356623e-06,
                1.31190777529513e-17,
                -1.35991280487585e-16,
                0.0258956622787094,
            ],
        ]
    )
    import time

    steps = 10000
    xs = []
    for j in range(3):
        obs, info = env.reset()
        for i in range(steps):
            action = -np.dot(K, obs[0:6])
            print(action)
            obs, reward, term, trunc, info = env.step(action)
            if term or steps - 1 == i:
                x = env.render()
                xs.append(x)
                break

        print(f"next {j+1}")
        time.sleep(1)
    env.close()
    for x in xs:
        plt.imshow(x)
        plt.pause(0.1)
    time.sleep(100)

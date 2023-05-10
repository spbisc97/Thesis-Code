import matplotlib as mpl


import matplotlib.pyplot as plt

import matplotlib.style as mplstyle

mplstyle.use("fast")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import jit
from numba.experimental import jitclass
import random
import time


# ? Maybe it's better to normalize the observation space?
# ? Maybe it's better to normalize the action space?
# ? Maybe it's better to use a discrete action space?
# ? Maybe it's better to use a continuous action space?
# ? Maybe it's better to use a continuous action space with a discrete action space?


class Satellite_rot(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "observation_spaces": ["MlpPolicy", "MultiInputPolicy"],
        "control": ["Ml", "ModelPredictiveControl", "LQR", "PID", "Random", "Human"],
        "action_spaces": ["continuous", "discrete"],
        "matplotlib_backend": ["TKAgg", "Qt5Agg", "WXAgg", "GTKAgg", "Qt4Agg"],
    }

    def __init__(
        self,
        render_mode=None,
        observation_space="MlpPolicy",
        action_space="continuous",
        control="Ml",
        matplotlib_backend="TKAgg",
    ):
        super(Satellite_rot, self).__init__()
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
        else:
            self.observation_space = spaces.Dict(
                {
                    "rot": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                    ),
                    "rot_speed": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                    ),
                }
            )
        # or render_mode in self.metadata['render_modes']
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode:
            mpl.use("TKAgg")

        self.qd = np.array([1, 0, 0, 0], dtype=np.float32)
        self.dmax = 20000
        self.vmax = 20000
        self.prev_shaping = None
        self.subplots = None

    def step(self, action):
        reward = 0
        truncated = False
        self.chaser.rotate(self._action_filter(action))
        terminated = self._beyond_observational_space()  # or self.chaser.fuel_mass <= 0
        observation = self._get_observation()
        info = self._get_info()
        reward = self._reward_function(action)

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._remember(observation, action, reward, info)
            if self.render_mode == "human":
                self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.chaser = Chaser()
        state = np.zeros((7,), dtype=np.float32)
        # state[random.randint(0,2)] = random.randint(-100, 100)
        # q0 = np.random.rand(1, 4) - 0.5
        (eul0,) = (np.random.rand(1, 3) - 0.5) * np.pi * 2
        q0 = eul_to_quat(eul0)
        state[0:4] = q0 / np.linalg.norm(q0)
        state[4:8] = (np.random.rand(1, 3) - 0.5) * 0
        self.chaser.set_state(state)
        self.prev_shaping = self._shape_reward()
        info = self._get_info()
        observation = self._get_observation()
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.statuses = np.array(self.chaser.state)
            self.euler_angles = np.array(quaternion_to_euler(self.chaser.state[0:4]))
            self.actions = np.array([]).reshape(0, 3)
            self.rewards = np.array([]).reshape(0, 1)
            self.rewards_sum = np.array([]).reshape(0, 1)
            self.infos = np.array(info)
            self.times = np.array([0])
        return observation, info

    def render(self):
        if self.render_mode == "human" and self.times[-1] % 20 != 0:
            plt.ion()
            return
        if self.subplots == None:
            fig, ax = plt.subplots(4, 3, figsize=(10, 10))
            lines = np.ma.zeros_like(ax)
            legend = np.array(
                [
                    ["roll", "pitch", "yaw"],
                    ["roll_speed", "pitch_speed", "yaw_speed"],
                    ["ux", "uy", "uz"],
                    ["reward", "reward_sum", ""],
                ]
            )
            limits = np.array(
                [
                    [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
                    [[-1, 1], [-1, 1], [-1, 1]],
                    [[-1, 1], [-1, 1], [-1, 1]],
                    [[-2, 0.3], [-100, 1], [-1, 1]],
                ]
            )
            (lines[0, 0],) = ax[0, 0].plot(self.times, self.euler_angles[:, 0])
            (lines[0, 1],) = ax[0, 1].plot(self.times, self.euler_angles[:, 1])
            (lines[0, 2],) = ax[0, 2].plot(self.times, self.euler_angles[:, 2])
            (lines[1, 0],) = ax[1, 0].plot(self.times, self.statuses[:, 4])
            (lines[1, 1],) = ax[1, 1].plot(self.times, self.statuses[:, 5])
            (lines[1, 2],) = ax[1, 2].plot(self.times, self.statuses[:, 6])
            (lines[2, 0],) = ax[2, 0].plot(self.times[:-1], self.actions[:, 0])
            (lines[2, 1],) = ax[2, 1].plot(self.times[:-1], self.actions[:, 1])
            (lines[2, 2],) = ax[2, 2].plot(self.times[:-1], self.actions[:, 2])
            (lines[3, 0],) = ax[3, 0].plot(self.times[:-1], self.rewards[:])
            (lines[3, 1],) = ax[3, 1].plot(self.times[:-1], self.rewards_sum[:])
            # (lines[3, 2],) = ax[3, 2].plot(self.times, self.rewards[:, 2])

            for idx, x in np.ndenumerate(ax):
                # ax[idx[0], idx[1]].set_xlim(0, 100)
                ax[idx[0], idx[1]].set_ylim(limits[idx[0], idx[1], :])

            for idx, x in np.ndenumerate(ax):
                ax[idx[0], idx[1]].set_title(legend[idx[0], idx[1]])

            plt.show(block=False)
            self.subplots = (fig, ax, lines)

        else:
            fig, ax, lines = self.subplots
            lines[0, 0].set_data(self.times, self.euler_angles[:, 0])
            lines[0, 1].set_data(self.times, self.euler_angles[:, 1])
            lines[0, 2].set_data(self.times, self.euler_angles[:, 2])
            lines[1, 0].set_data(self.times, self.statuses[:, 4])
            lines[1, 1].set_data(self.times, self.statuses[:, 5])
            lines[1, 2].set_data(self.times, self.statuses[:, 6])
            lines[2, 0].set_data(self.times[:-1], self.actions[:, 0])
            lines[2, 1].set_data(self.times[:-1], self.actions[:, 1])
            lines[2, 2].set_data(self.times[:-1], self.actions[:, 2])
            lines[3, 0].set_data(self.times[:-1], self.rewards[:])
            lines[3, 1].set_data(self.times[:-1], self.rewards_sum[:])

        mplstyle.use("fast")

        for idx, x in np.ndenumerate(ax):
            ax[idx[0], idx[1]].relim(visible_only=True)
            ax[idx[0], idx[1]].autoscale(enable=True, axis="x", tight=True)

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
            return np.array(fig.canvas.renderer.buffer_rgba())
        # if self.render_mode == "human":
        # fig.show()

        return

    def close(self):
        return super().close()

    def _remember(self, sta, act, rew, inf={}):
        self.statuses = np.vstack((self.statuses, sta))
        self.euler_angles = np.vstack((self.euler_angles, quaternion_to_euler(sta[:4])))
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
                    # we could also pass just the error(quaternion) instead of the state
                    # till qd = (1 0 0 0), is exactly the same
                )
            )
        else:
            return {
                "rot": self.chaser.state[:4],
                "rot_speed": self.chaser.state[4:8],
            }

    def _reward_function(self, action):
        reward = 0
        # distance_reward = prev_distance - distance
        # fuel_reward = prev_fuel - fuel

        shaping = self._shape_reward()

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Attitude Error Term
        attitude_error_term = -np.linalg.norm(
            self.chaser.quat_track_err(self.chaser.state[:4], self.qd)[1:]
        )

        # Control Effort Term
        control_effort_term = -0.1 * np.linalg.norm(action)

        # Stability Term
        stability_term = -0.1 * np.linalg.norm(self.chaser.state[4:8])

        # Smoothness Term
        # smoothness_term = -0.001 * np.linalg.norm(np.gradient(angular_velocity))

        # Total Reward
        reward = (
            attitude_error_term
            + control_effort_term
            + stability_term
            #  + smoothness_term
        )

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

    def _get_info(self):
        return {"qd": self.qd}


class Chaser:
    mass = 30
    initial_battery = 10
    # mu = 3.986004418 * 1e14
    # rt = 6.6 * 1e6
    Tmax = 1.05e-2
    g = 9.81
    dt = 0.5
    I = np.diag([8.33e-2, 1.08e-1, 4.17e-2])
    invI = np.diag([12.0048019207683, 9.25925925925926, 23.9808153477218])

    def __init__(self, step=1):
        # self.position = np.array([0,0,0,0])
        # self.velocity = np.array([0,0,0])
        self.state = np.zeros((7,), dtype=np.float32)
        self.invI = np.linalg.inv(self.I)

        self.step = self.dt if step is None else step

    def set_state(self, state):
        self.state = np.float32(state)
        return self.state

    def _Sat_Rotational_Dyn(self, y, t, trust):
        dy = np.zeros(
            7,
        )
        q = y[0:4]
        w = y[4:8]

        torques = trust.copy()  # + noise

        ep = 1 - np.dot(q, q)
        K = 0.1
        dy[0:4] = 0.5 * self.omega(np.hstack((0, w))) @ q + K * ep * q
        # q = q / np.linalg.norm(q) to keep norm=1
        dy[4:] = self.invI @ (torques - np.cross(w, np.dot(self.I, w)))

        return dy

    @staticmethod
    @jit(nopython=True)
    def omega(q):
        mat = np.array(
            [
                [q[0], -q[1], -q[2], -q[3]],
                [q[1], q[0], q[3], -q[2]],
                [q[2], -q[3], q[0], q[1]],
                [q[3], q[2], -q[1], q[0]],
            ]
        )
        return mat

    def rotate(self, trust):
        trust = np.clip(trust, -Chaser.Tmax, Chaser.Tmax)
        self.state = self._rk4(
            self._Sat_Rotational_Dyn, self.state, self.dt, args=(trust,)
        )
        pass

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

    @staticmethod
    def quaternion_err_rate(q1, qd, w=np.array([0, 0, 0]), wd=np.array([0, 0, 0])):
        q_e = Chaser.quat_track_err(q1, qd)
        p = -q_e[1:4] / (1 - np.dot(q_e[1:4], q_e[1:4]))
        d = wd - w
        kp = 0.01 * Chaser.invI
        kd = 0.05 * Chaser.invI
        u = kp @ p + kd @ d
        return u

    @staticmethod
    def quat_track_err(q1, qd):
        q_inv = Chaser.quat_inv(qd)
        e = Chaser.omega(q_inv) @ q1  # equal to quaternion_multiply(q1,qd_inv)):
        if np.linalg.norm(e) > 1.1:
            print("error_norm_error")
            e = e / np.linalg.norm(e)
        return e

    @staticmethod
    @jit(nopython=True)
    def quat_inv(q):
        # shouldnt be needed the denominator is always 1
        den = np.dot(q, q)
        return np.array([q[0], -q[1], -q[2], -q[3]]) / den


# i could use abr_control
@jit(nopython=True)
def eul_to_quat(eul):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll, pitch, yaw = eul
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


@jit(nopython=True)
def quaternion_to_euler(q):
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [yaw, pitch, roll]


if __name__ == "__main__":
    # Q_main()
    from stable_baselines3.common.env_checker import check_env

    check_env(Satellite_rot(), warn=True)
    print("env checked")
    env = Satellite_rot(render_mode="human")
    obs, info = env.reset()
    t_start = time.time()
    for i in range(600):
        action = Chaser.quaternion_err_rate(obs[0:4], info["qd"], w=obs[4:8])
        # action = np.array([0, 0, 0])
        obs, reward, term, trunc, info = env.step(action)
        # if np.linalg.norm(obs[4:8]) < 0.001:
        # break
        if term:
            break
    print(time.time() - t_start)
    env.close()

    env = Satellite_rot(render_mode="rgb_array")
    term = False
    obs, info = env.reset()
    rwds = 0
    t_start = time.time()
    steps = 10000
    for i in range(10000):
        action = Chaser.quaternion_err_rate(obs[0:4], info["qd"], w=obs[4:8])
        # action = np.array([0, 0, 0])
        obs, reward, term, trunc, info = env.step(action)
        rwds += reward
        if i == steps - 1:
            term = True
        if term:
            plt.imshow(env.render())
            time.sleep(5)
            break
    print(time.time() - t_start)
    env.close()

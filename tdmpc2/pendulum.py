__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn



DEFAULT_X = np.pi
DEFAULT_Y = 1.0

class FrameSkip(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class PendulumEnv(gym.Env):
    """
    ## Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: .

    Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
    the acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
    The default value is `g = 10.0`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5
    (array([ 0.4123625 ,  0.91101986, -0.89235795], dtype=float32), {})

    ```

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.obs_rad = 0.25

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        # self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(500, 500, 3), dtype=np.uint8
        )

    def step(self, u):
        self.ctime_step += 1
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        # self.obs_vel[1] = self.obs_vel[1] - self.g * dt
        self.obs_state = self.obs_state + self.obs_vel * dt

        if self.render_mode == "human":
            self.render()
        
        if self.check_collision():
            return self._get_obs(), -100, True, False, {}
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.ctime_step = 0
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.state[0] = -3*np.pi/4
        colliding = True
        while colliding:
            self.obs_state = self.np_random.uniform(low=-self.l, high=self.l, size=(2,))
            colliding = self.check_collision()
        self.obs_vel = self.np_random.uniform(low=-self.max_speed, high=self.max_speed, size=(2,))
        self.obs_vel = np.zeros(2)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def check_collision(self):
        # Create numpy arrays for the segment points and circle center
        p1 = np.array([0, 0])
        theta = self.state[0] + np.pi / 2
        p2 = np.array([self.l*np.cos(theta), self.l*np.sin(theta)])
        
        center = np.array([self.obs_state[0], self.obs_state[1]])
        # Calculate the distance from the line segment to the center of the circle
        diff = p2 - p1
        t = np.dot(center - p1, diff) / np.dot(diff, diff)
        t = np.clip(t, 0, 1)  # Ensure t is between 0 and 1 to stay within the segment
        closest = p1 + t * diff  # This is the point on the line segment closest to the center of the circle
        distance = np.linalg.norm(center - closest)

        # Check for collision
        return distance <= self.obs_rad

    def _get_obs(self):           
        # theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        old_render = self.render_mode
        self.render_mode = "rgb_array"
        obs = self.render()
        self.render_mode = old_render
        return obs

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()


        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.02 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width * 2), (204, 77, 77)
        )

        # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        # img = pygame.image.load(fname)
        # if self.last_u is not None:
        #     scale_img = pygame.transform.smoothscale(
        #         img,
        #         (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
        #     )
        #     is_flip = bool(self.last_u > 0)
        #     scale_img = pygame.transform.flip(scale_img, is_flip, True)
        #     self.surf.blit(
        #         scale_img,
        #         (
        #             offset - scale_img.get_rect().centerx,
        #             offset - scale_img.get_rect().centery,
        #         ),
        #     )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        # drawing a new circle
        circle_radius = int(self.obs_rad * scale)
        circle_position = (int(offset + self.obs_state[0] * scale), int(offset + self.obs_state[1] * scale))
        gfxdraw.aacircle(self.surf, circle_position[0], circle_position[1], circle_radius, (0, 255, 0))
        gfxdraw.filled_circle(self.surf, circle_position[0], circle_position[1], circle_radius, (0, 255, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces, Env
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

class MyFrameStack(Env):
    """
    Frame stacking wrapper for a single environment. Designed for image observations.

    :param env: Environment to wrap
    :param n_stack: Number of frames to stack
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
        Alternatively channels_order can be a dictionary which can be used with environments with Dict observation spaces
    """
    def __init__(self, env: Env, n_stack: int, channels_order: Optional[Union[str, Mapping[str, str]]] = None) -> None:
        assert isinstance(
            env.observation_space, (spaces.Box, spaces.Dict)
        ), "FrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces"

        self.stacked_obs = StackedObservations(1, n_stack, env.observation_space, channels_order)
        self.observation_space = self.stacked_obs.stacked_observation_space
        self.env = env

    def step(self, action) -> Tuple[
        Union[np.ndarray, Dict[str, np.ndarray]],
        np.ndarray,
        np.ndarray,
        Dict[str, Any],
    ]:
        observation, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        observation = self.stacked_obs.update(observation, np.array([0, done]), [info])
        return observation[0], reward, done, info

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset the environment
        """
        observation = self.env.reset()
        observation = self.stacked_obs.reset(observation[0])
        return observation

    def render(self):
        self.env.render()

if __name__ == "__main__":
    def make_env():
        env = PendulumEnv(render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=200)
        env = ResizeObservation(env, shape=(64, 64))
        env = GrayScaleObservation(env, keep_dim=True)
        env = FrameSkip(env, skip=2)
        return env
    
    from functools import partial
    env_fn = partial(make_env)
    vec_env = make_vec_env(env_fn, n_envs=8, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    ppo_args = {}
    ppo_args["batch_size"] = 128
    ppo_args["n_steps"] = 512
    eval_freq = ppo_args["n_steps"] + 1
    ppo_args["gamma"] = 0.99
    ppo_args["gae_lambda"] = 0.95
    ppo_args["n_epochs"] = 10
    ppo_args["ent_coef"] = 0.0
    ppo_args["sde_sample_freq"] = 4
    ppo_args["max_grad_norm"] = 0.5
    ppo_args["vf_coef"] = 0.5
    ppo_args["learning_rate"] = 1e-4
    ppo_args["use_sde"] = True
    ppo_args["clip_range"] = 0.2
    ppo_args["policy_kwargs"] = dict(log_std_init=-2,
                        ortho_init=False,
                        activation_fn=nn.GELU,
                        net_arch=dict(pi=[256], vf=[256]),
                        )
    model = PPO("CnnPolicy", vec_env, tensorboard_log='sb3/logs/pendulum', verbose=0, **ppo_args)
    eval_callback = EvalCallback(eval_env, best_model_save_path='sb3/pendulum', log_path='sb3/pendulum', eval_freq=eval_freq, deterministic=True)
    model.learn(total_timesteps=4_000_000, progress_bar=True, callback=eval_callback)
    model.save("sb3/ppo_pendulum")

    # model = PPO.load("sb3/pendulum/best_model.zip")
    # done = False
    # env = make_env()
    # # env = MyFrameStack(env, n_stack=2)
    # obs, _ = env.reset()
    # stacked_obs = np.concatenate([obs, obs], axis=-1)
    # ret = 0
    # while not done:
    #     action = model.predict(stacked_obs, deterministic=True)
    #     obs, rew, term, trunc, info = env.step(action[0])
    #     done = term or trunc
    #     ret += rew
    #     stacked_obs[:, :, 0] = stacked_obs[:, :, 1]
    #     stacked_obs[:, :, 1] = obs.squeeze()
    #     print(ret)
    #     env.render()
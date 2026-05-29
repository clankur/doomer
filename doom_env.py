"""ViZDoom environment wrapper with standard preprocessing for RL."""

from dataclasses import dataclass
from typing import Optional

import gymnasium
import numpy as np
import vizdoom.gymnasium_wrapper  # noqa: F401 — registers ViZDoom envs
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation, TimeLimit


@dataclass(frozen=True)
class EnvConfig:
    scenario: str = "VizdoomBasic-v1"
    frame_skip: int = 4
    frame_stack: int = 4
    resolution: int = 84
    max_episode_steps: Optional[int] = 300
    dense_reward: bool = False
    ammo_cost: float = 0.1
    living_penalty: float = 0.01


class ExtractScreen(gymnasium.ObservationWrapper):
    """ViZDoom returns Dict(screen=..., gamevariables=...). Extract just the screen."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs):
        return obs["screen"]


class NormalizePixels(gymnasium.ObservationWrapper):
    """Scale uint8 [0,255] observations to float32 [0,1]."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32) / 255.0


class DenseRewardWrapper(gymnasium.Wrapper):
    """Shape rewards using per-step game variable deltas.

    VizdoomBasic-v1 exposes AMMO2 as gamevariables[0].
    """

    def __init__(self, env: gymnasium.Env, ammo_cost: float = 0.1, living_penalty: float = 0.01):
        super().__init__(env)
        self.ammo_cost = ammo_cost
        self.living_penalty = living_penalty
        self._prev_ammo: float = 0.0

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self._prev_ammo = float(obs["gamevariables"][0])
        return obs, info

    def step(self, action) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        ammo: float = float(obs["gamevariables"][0])
        ammo_used: float = self._prev_ammo - ammo  # positive when a shot was fired
        shaped: float = reward - self.ammo_cost * ammo_used - self.living_penalty
        self._prev_ammo = ammo
        return obs, shaped, terminated, truncated, info


def make_env(config: EnvConfig, seed: int = 0) -> gymnasium.Env:
    """Create a single preprocessed ViZDoom environment."""
    env = gymnasium.make(config.scenario, frame_skip=config.frame_skip)
    if config.dense_reward:
        env = DenseRewardWrapper(env, ammo_cost=config.ammo_cost, living_penalty=config.living_penalty)
    env = ExtractScreen(env)
    if config.max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=(config.resolution, config.resolution))
    env = NormalizePixels(env)
    env = FrameStackObservation(env, stack_size=config.frame_stack)
    env.reset(seed=seed)
    return env

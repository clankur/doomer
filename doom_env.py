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


def make_env(config: EnvConfig, seed: int = 0) -> gymnasium.Env:
    """Create a single preprocessed ViZDoom environment."""
    env = gymnasium.make(config.scenario, frame_skip=config.frame_skip)
    env = ExtractScreen(env)
    if config.max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=(config.resolution, config.resolution))
    env = NormalizePixels(env)
    env = FrameStackObservation(env, stack_size=config.frame_stack)
    env.reset(seed=seed)
    return env

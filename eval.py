"""Watch a trained agent play Doom.

Usage:
    uv run python eval.py --checkpoint /tmp/smoke_000/checkpoint_25.pt --config-name=local_test
    uv run python eval.py --episodes 5 --config-name=local_test
"""

import gymnasium
import hydra
import numpy as np
import torch
import vizdoom.gymnasium_wrapper  # noqa: F401
from einops import rearrange
from omegaconf import DictConfig

from doom_env import EnvConfig, ExtractScreen, NormalizePixels
from train import PolicyNetwork, build_config


def make_render_env(config: EnvConfig) -> gymnasium.Env:
    """Like make_env but with human-visible rendering."""
    env = gymnasium.make(config.scenario, frame_skip=config.frame_skip, render_mode="human")
    env = ExtractScreen(env)
    from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation, TimeLimit

    if config.max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, shape=(config.resolution, config.resolution))
    env = NormalizePixels(env)
    env = FrameStackObservation(env, stack_size=config.frame_stack)
    return env


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    config = build_config(cfg)
    device = torch.device("cpu")

    env = make_render_env(config.env)
    num_actions = env.action_space.n

    policy = PolicyNetwork(
        frame_stack=config.env.frame_stack,
        num_actions=num_actions,
        resolution=config.env.resolution,
    )

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint["policy"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint provided — running with random policy")

    policy.eval()
    episodes = cfg.get("episodes", 3)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_t = rearrange(torch.from_numpy(np.array(obs)), "frames h w -> 1 frames h w")
            with torch.no_grad():
                dist = policy(obs_t)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep + 1}: return={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()

"""Doom RL training: network, episode collection, REINFORCE, Hydra entrypoint.

Usage:
    uv run python -m train --config-name=local_test ++paths.model_name=smoke_000
"""

import os
from dataclasses import dataclass

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from doom_env import EnvConfig, make_env

# ─── Config dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingHparams:
    algorithm: str = "reinforce"
    learning_rate: float = 1e-4
    gamma: float = 0.99
    max_episodes: int = 5000
    log_interval: int = 10
    checkpoint_interval: int = 500
    seed: int = 42


@dataclass(frozen=True)
class Paths:
    root_working_dir: str = "/tmp"
    model_name: str = "default"


@dataclass(frozen=True)
class Config:
    env: EnvConfig = None
    training: TrainingHparams = None
    paths: Paths = None


def build_config(cfg: DictConfig) -> Config:
    return Config(
        env=EnvConfig(**cfg.env),
        training=TrainingHparams(**cfg.training),
        paths=Paths(**cfg.paths),
    )


# ─── Network ────────────────────────────────────────────────────────────────


class PolicyNetwork(nn.Module):
    """Nature DQN CNN backbone + policy head.

    Input: (B, frame_stack, H, W) float32 in [0, 1]
    Output: Categorical distribution over actions
    """

    def __init__(self, frame_stack: int, num_actions: int, resolution: int = 84):
        super().__init__()
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute conv output size dynamically so any resolution works
        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack, resolution, resolution)
            conv_out = self._conv_forward(dummy)
            conv_flat = conv_out.numel()

        self.fc = nn.Linear(conv_flat, 512)
        self.policy_head = nn.Linear(512, num_actions)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        x = self._conv_forward(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.policy_head(x)
        return torch.distributions.Categorical(logits=logits)


# ─── REINFORCE ──────────────────────────────────────────────────────────────


def collect_episode(env, policy, device):
    """Roll out one full episode, collecting log-probs and rewards."""
    obs, _ = env.reset()
    log_probs = []
    rewards = []
    done = False

    while not done:
        obs_t = torch.from_numpy(np.array(obs)).unsqueeze(0).to(device)
        dist = policy(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        obs, reward, terminated, truncated, _info = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)

    return log_probs, rewards


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Discounted returns: G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)


def reinforce_loss(log_probs: list[torch.Tensor], returns: torch.Tensor) -> torch.Tensor:
    """REINFORCE policy gradient with mean-return baseline."""
    # Subtracting mean return as baseline reduces variance without introducing bias
    advantages = returns - returns.mean()
    if advantages.std() > 1e-8:
        advantages = advantages / advantages.std()

    loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()
    return loss


# ─── Training loop ──────────────────────────────────────────────────────────


def train_reinforce(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.training.seed)

    env = make_env(config.env, seed=config.training.seed)
    num_actions = env.action_space.n

    policy = PolicyNetwork(
        frame_stack=config.env.frame_stack,
        num_actions=num_actions,
        resolution=config.env.resolution,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.training.learning_rate)

    model_dir = os.path.join(config.paths.root_working_dir, config.paths.model_name)
    os.makedirs(model_dir, exist_ok=True)

    running_reward = None

    for episode in range(config.training.max_episodes):
        log_probs, rewards = collect_episode(env, policy, device)
        returns = compute_returns(rewards, config.training.gamma)

        loss = reinforce_loss(log_probs, returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(rewards)
        running_reward = episode_return if running_reward is None else 0.05 * episode_return + 0.95 * running_reward

        if episode % config.training.log_interval == 0:
            print(
                f"[Episode {episode:>5d}] "
                f"return={episode_return:>7.1f}  "
                f"running={running_reward:>7.1f}  "
                f"length={len(rewards):>4d}  "
                f"loss={loss.item():>8.4f}"
            )

        if (
            config.training.checkpoint_interval > 0
            and episode > 0
            and episode % config.training.checkpoint_interval == 0
        ):
            path = os.path.join(model_dir, f"checkpoint_{episode}.pt")
            torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "episode": episode}, path)
            print(f"  → saved {path}")

    env.close()
    print(f"Training complete. Final running reward: {running_reward:.1f}")


# ─── Entrypoint ─────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig):
    config = build_config(cfg)

    if config.training.algorithm == "reinforce":
        train_reinforce(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.training.algorithm}")


if __name__ == "__main__":
    main()

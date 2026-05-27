"""Doom RL training: network, episode collection, REINFORCE, Hydra entrypoint.

Usage:
    uv run python -m train --config-name=local_test ++paths.model_name=smoke_000
"""

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from doom_env import EnvConfig, make_env

try:
    import runq
except ImportError:
    runq = None

try:
    import wandb
except ImportError:
    wandb = None

# ─── Config dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingHparams:
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
        # x: (batch, frames, h, w)
        x = self._conv_forward(x)  # -> (batch, channels, h', w')
        x = rearrange(x, "batch channels h w -> batch (channels h w)")
        x = F.relu(self.fc(x))  # -> (batch, hidden)
        logits = self.policy_head(x)  # -> (batch, actions)
        return torch.distributions.Categorical(logits=logits)


# ─── Episode collection & loss ──────────────────────────────────────────────


@dataclass
class EpisodeResult:
    log_probs: list[torch.Tensor]
    rewards: list[float]
    episode_return: float
    length: int


def collect_episode(env: gymnasium.Env, policy: PolicyNetwork, device: torch.device) -> EpisodeResult:
    """Roll out one full episode, collecting log-probs and rewards."""
    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    done = False

    while not done:
        obs_t = rearrange(torch.from_numpy(np.array(obs)), "frames h w -> 1 frames h w").to(device)
        dist = policy(obs_t)  # -> Categorical over (1, actions)
        action = dist.sample()  # -> (1,)
        log_prob = dist.log_prob(action)  # -> (1,)

        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(float(reward))

    return EpisodeResult(
        log_probs=log_probs,
        rewards=rewards,
        episode_return=sum(rewards),
        length=len(rewards),
    )


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Discounted returns: G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}."""
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns)


def loss(log_probs: list[torch.Tensor], returns: torch.Tensor) -> torch.Tensor:
    """REINFORCE policy gradient with mean-return baseline."""
    # Subtracting mean return as baseline reduces variance without introducing bias

    # note: in my own literature, reccomendations for updating REINFORCE was to
    #   use critic network for baseline centering (no std normalization)
    #   GAE/TD error to estimate advantage

    advantages = returns - returns.mean()
    if advantages.std() > 1e-8:
        advantages = advantages / advantages.std()

    # log_probs: T x (1,) tensors, advantages: (T,)
    stacked = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)])  # -> (T, 1)
    return rearrange(stacked, "t 1 -> t").sum()


# ─── Training loop ──────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    episode: int
    episode_return: float
    running_reward: float
    episode_length: int
    loss: float
    elapsed_sec: float


def _report_wandb_url_to_runq(wandb_url: str) -> None:
    """If running inside runq, report the wandb URL back via the PATCH API."""
    runq_server = os.environ.get("RUNQ_SERVER")
    experiment_id = os.environ.get("RUNQ_EXPERIMENT_ID")
    if not runq_server or not experiment_id:
        return
    try:
        data = json.dumps({"wandb_url": wandb_url}).encode()
        req = urllib.request.Request(
            f"{runq_server}/api/experiments/{experiment_id}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def train(config: Config) -> None:
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

    # Save config for reproducibility
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    if wandb is not None:
        wandb.init(project="doomer", name=config.paths.model_name, config=asdict(config))
        _report_wandb_url_to_runq(wandb.run.get_url())

    metrics_log: list[StepMetrics] = []
    running_reward: float = 0.0
    t_start = time.time()

    for episode in range(config.training.max_episodes):
        result = collect_episode(env, policy, device)
        returns = compute_returns(result.rewards, config.training.gamma)

        step_loss = loss(result.log_probs, returns)

        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()

        running_reward = (
            result.episode_return if episode == 0 else 0.05 * result.episode_return + 0.95 * running_reward
        )

        metrics = StepMetrics(
            episode=episode,
            episode_return=result.episode_return,
            running_reward=running_reward,
            episode_length=result.length,
            loss=step_loss.item(),
            elapsed_sec=time.time() - t_start,
        )
        metrics_log.append(metrics)

        if episode % config.training.log_interval == 0:
            print(
                f"[Episode {episode:>5d}] "
                f"return={metrics.episode_return:>7.1f}  "
                f"running={metrics.running_reward:>7.1f}  "
                f"length={metrics.episode_length:>4d}  "
                f"loss={metrics.loss:>8.4f}  "
                f"t={metrics.elapsed_sec:>6.1f}s"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "episode_return": metrics.episode_return,
                        "running_reward": metrics.running_reward,
                        "episode_length": metrics.episode_length,
                        "loss": metrics.loss,
                        "policy_loss": metrics.policy_loss,
                        "value_loss": metrics.value_loss,
                        "entropy": metrics.entropy,
                    },
                    step=metrics.episode,
                )

        if (
            config.training.checkpoint_interval > 0
            and episode > 0
            and episode % config.training.checkpoint_interval == 0
        ):
            path = os.path.join(model_dir, f"checkpoint_{episode}.pt")
            torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "episode": episode}, path)
            print(f"  → saved {path}")

    # Write metrics log for post-hoc analysis
    metrics_path = os.path.join(model_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(asdict(m)) + "\n")

    env.close()
    if wandb is not None:
        wandb.finish()
    print(f"Training complete. Final running reward: {running_reward:.1f}")
    print(f"Metrics saved to {metrics_path}")


# ─── Entrypoint ─────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    config = build_config(cfg)
    if runq is not None:
        task = runq.Task(project="doomer", name=config.paths.model_name)
        task.execute_remotely(queue="gpu")
    train(config)


if __name__ == "__main__":
    main()

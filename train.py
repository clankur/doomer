"""Doom RL training: policy network, GRPO with group-relative advantages, Hydra entrypoint.

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
import runq
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

import wandb
from doom_env import EnvConfig, make_env

# ─── Config dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingHparams:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    max_episodes: int = 5000
    log_interval: int = 10
    checkpoint_interval: int = 500
    seed: int = 42
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    group_size: int = 8


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
        x = self._conv_forward(x)  # -> (batch, channels, h', w')
        x = rearrange(x, "batch channels h w -> batch (channels h w)")
        x = F.relu(self.fc(x))  # -> (batch, hidden)
        logits = self.policy_head(x)  # -> (batch, actions)
        return torch.distributions.Categorical(logits=logits)


# ─── Episode collection & GRPO loss ─────────────────────────────────────────


@dataclass
class EpisodeData:
    observations: torch.Tensor  # (T, frames, H, W)
    actions: torch.Tensor  # (T,)
    old_log_probs: torch.Tensor  # (T,)
    episode_return: float
    length: int


@dataclass
class GroupData:
    observations: torch.Tensor  # (total_T, frames, H, W)
    actions: torch.Tensor  # (total_T,)
    old_log_probs: torch.Tensor  # (total_T,)
    advantages: torch.Tensor  # (total_T,)
    episode_returns: list[float]
    episode_lengths: list[int]
    total_timesteps: int


def collect_episode(env: gymnasium.Env, policy: PolicyNetwork, device: torch.device) -> EpisodeData:
    """Roll out one full episode under the current policy."""
    obs, _ = env.reset()
    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    log_prob_list: list[torch.Tensor] = []
    reward_total: float = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs_np = np.array(obs)
        obs_list.append(obs_np)

        obs_t = rearrange(torch.from_numpy(obs_np), "frames h w -> 1 frames h w").to(device)
        with torch.no_grad():
            dist = policy(obs_t)
        action = dist.sample()  # -> (1,)
        log_prob = dist.log_prob(action)  # -> (1,)

        obs, reward, terminated, truncated, _ = env.step(action.item())

        action_list.append(action.item())
        log_prob_list.append(log_prob.cpu())
        reward_total += float(reward)

    observations = torch.stack([torch.from_numpy(o) for o in obs_list])  # (T, frames, H, W)
    actions = torch.tensor(action_list, dtype=torch.long)  # (T,)
    old_log_probs = rearrange(torch.stack(log_prob_list), "t 1 -> t")  # (T,)

    return EpisodeData(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        episode_return=reward_total,
        length=len(action_list),
    )


def collect_group(
    env: gymnasium.Env,
    policy: PolicyNetwork,
    device: torch.device,
    group_size: int,
) -> GroupData:
    """Collect a group of episodes and compute group-relative advantages."""
    episodes = [collect_episode(env, policy, device) for _ in range(group_size)]

    returns = torch.tensor([ep.episode_return for ep in episodes])
    group_mean = returns.mean().item()
    group_std = returns.std().item()

    advantages = torch.cat(
        [torch.full((ep.length,), (ep.episode_return - group_mean) / (group_std + 1e-8)) for ep in episodes]
    )

    return GroupData(
        observations=torch.cat([ep.observations for ep in episodes]),
        actions=torch.cat([ep.actions for ep in episodes]),
        old_log_probs=torch.cat([ep.old_log_probs for ep in episodes]),
        advantages=advantages,
        episode_returns=[ep.episode_return for ep in episodes],
        episode_lengths=[ep.length for ep in episodes],
        total_timesteps=sum(ep.length for ep in episodes),
    )


def grpo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    clip_epsilon: float,
    entropy_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GRPO clipped surrogate objective with entropy bonus.

    All inputs are (MB,) tensors (minibatch).
    Returns (total_loss, policy_loss, entropy) as scalars.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)  # (MB,)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    entropy_mean = entropy.mean()
    total_loss = policy_loss - entropy_coeff * entropy_mean
    return total_loss, policy_loss, entropy_mean


# ─── Training loop ──────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    group: int
    group_mean_return: float
    group_std_return: float
    group_min_return: float
    group_max_return: float
    running_reward: float
    mean_episode_length: float
    total_timesteps: int
    loss: float
    policy_loss: float
    entropy: float
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

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    wandb_mode = "online" if os.environ.get("RUNQ_EXPERIMENT_ID", "local") != "local" else "disabled"
    wandb.init(project="doomer", name=config.paths.model_name, config=asdict(config), mode=wandb_mode)
    _report_wandb_url_to_runq(wandb.run.get_url())

    metrics_log: list[StepMetrics] = []
    running_reward: float = 0.0
    total_episodes: int = 0
    t_start = time.time()

    num_groups = config.training.max_episodes // config.training.group_size

    for group_idx in range(num_groups):
        group = collect_group(env, policy, device, config.training.group_size)
        total_episodes += config.training.group_size

        obs_device = group.observations.to(device)  # (total_T, frames, H, W)
        actions_device = group.actions.to(device)  # (total_T,)
        old_lp_device = group.old_log_probs.to(device)  # (total_T,)
        adv_device = group.advantages.to(device)  # (total_T,)

        T = group.total_timesteps
        epoch_total: list[float] = []
        epoch_policy: list[float] = []
        epoch_entropy: list[float] = []

        for _epoch in range(config.training.update_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, config.training.minibatch_size):
                mb_idx = indices[start : start + config.training.minibatch_size]

                mb_obs = obs_device[mb_idx]  # (MB, frames, H, W)
                mb_actions = actions_device[mb_idx]  # (MB,)
                mb_old_lp = old_lp_device[mb_idx]  # (MB,)
                mb_adv = adv_device[mb_idx]  # (MB,)

                dist = policy(mb_obs)
                new_log_probs = dist.log_prob(mb_actions)  # (MB,)
                entropy = dist.entropy()  # (MB,)

                total, p_loss, ent = grpo_loss(
                    old_log_probs=mb_old_lp,
                    new_log_probs=new_log_probs,
                    advantages=mb_adv,
                    entropy=entropy,
                    clip_epsilon=config.training.clip_epsilon,
                    entropy_coeff=config.training.entropy_coeff,
                )

                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.training.max_grad_norm)
                optimizer.step()

                epoch_total.append(total.item())
                epoch_policy.append(p_loss.item())
                epoch_entropy.append(ent.item())

        group_returns = torch.tensor(group.episode_returns)
        group_mean = group_returns.mean().item()
        group_std = group_returns.std().item()
        running_reward = group_mean if group_idx == 0 else 0.05 * group_mean + 0.95 * running_reward

        metrics = StepMetrics(
            group=group_idx,
            group_mean_return=group_mean,
            group_std_return=group_std,
            group_min_return=min(group.episode_returns),
            group_max_return=max(group.episode_returns),
            running_reward=running_reward,
            mean_episode_length=group.total_timesteps / config.training.group_size,
            total_timesteps=group.total_timesteps,
            loss=sum(epoch_total) / len(epoch_total),
            policy_loss=sum(epoch_policy) / len(epoch_policy),
            entropy=sum(epoch_entropy) / len(epoch_entropy),
            elapsed_sec=time.time() - t_start,
        )
        metrics_log.append(metrics)

        if group_idx % config.training.log_interval == 0:
            print(
                f"[Group {group_idx:>4d}] "
                f"mean_ret={metrics.group_mean_return:>7.1f}  "
                f"running={metrics.running_reward:>7.1f}  "
                f"std={metrics.group_std_return:>5.2f}  "
                f"min={metrics.group_min_return:>5.1f}  "
                f"max={metrics.group_max_return:>5.1f}  "
                f"loss={metrics.loss:>8.4f}  "
                f"entropy={metrics.entropy:>6.4f}  "
                f"t={metrics.elapsed_sec:>6.1f}s",
                flush=True,
            )
            wandb.log(
                {
                    "group_mean_return": metrics.group_mean_return,
                    "group_std_return": metrics.group_std_return,
                    "group_min_return": metrics.group_min_return,
                    "group_max_return": metrics.group_max_return,
                    "running_reward": metrics.running_reward,
                    "mean_episode_length": metrics.mean_episode_length,
                    "loss": metrics.loss,
                    "policy_loss": metrics.policy_loss,
                    "entropy": metrics.entropy,
                },
                step=total_episodes,
            )

        if (
            config.training.checkpoint_interval > 0
            and group_idx > 0
            and group_idx % config.training.checkpoint_interval == 0
        ):
            path = os.path.join(model_dir, f"checkpoint_{group_idx}.pt")
            torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "group": group_idx}, path)
            print(f"  → saved {path}", flush=True)

    final_path = os.path.join(model_dir, "checkpoint_final.pt")
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "group": group_idx}, final_path)
    print(f"  → saved {final_path}", flush=True)

    metrics_path = os.path.join(model_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(asdict(m)) + "\n")

    env.close()
    wandb.finish()
    print(f"Training complete. Final running reward: {running_reward:.1f}")
    print(f"Metrics saved to {metrics_path}")


# ─── Entrypoint ─────────────────────────────────────────────────────────────


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    config = build_config(cfg)
    hydra_overrides = [o.split("=")[0].lstrip("+") for o in HydraConfig.get().overrides.task]
    task = runq.Task(project="doomer", name=config.paths.model_name)
    task.execute_remotely(queue="gpu", config=OmegaConf.to_yaml(cfg), overrides=hydra_overrides)
    train(config)


if __name__ == "__main__":
    main()

"""Doom RL training: actor-critic network, PPO with GAE, Hydra entrypoint.

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
import wandb
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

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
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatch_size: int = 64


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


class ActorCriticNetwork(nn.Module):
    """Nature DQN CNN backbone + policy and value heads.

    Input: (B, frame_stack, H, W) float32 in [0, 1]
    Output: (Categorical distribution over actions, value estimates)
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
        self.value_head = nn.Linear(512, 1)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        # x: (batch, frames, h, w)
        x = self._conv_forward(x)  # -> (batch, channels, h', w')
        x = rearrange(x, "batch channels h w -> batch (channels h w)")
        x = F.relu(self.fc(x))  # -> (batch, hidden)
        logits = self.policy_head(x)  # -> (batch, actions)
        value = self.value_head(x)  # -> (batch, 1)
        value = rearrange(value, "batch 1 -> batch")
        return torch.distributions.Categorical(logits=logits), value


# ─── Rollout collection & PPO loss ─────────────────────────────────────────


@dataclass
class RolloutData:
    observations: torch.Tensor  # (T, frames, H, W)
    actions: torch.Tensor  # (T,)
    old_log_probs: torch.Tensor  # (T,)
    rewards: list[float]
    values: torch.Tensor  # (T,)
    last_value: float
    episode_return: float
    length: int


def collect_rollout(env: gymnasium.Env, policy: ActorCriticNetwork, device: torch.device) -> RolloutData:
    """Roll out one full episode, collecting everything PPO needs."""
    obs, _ = env.reset()
    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    log_prob_list: list[torch.Tensor] = []
    reward_list: list[float] = []
    value_list: list[torch.Tensor] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs_np = np.array(obs)
        obs_list.append(obs_np)

        obs_t = rearrange(torch.from_numpy(obs_np), "frames h w -> 1 frames h w").to(device)
        with torch.no_grad():
            dist, value = policy(obs_t)
        action = dist.sample()  # -> (1,)
        log_prob = dist.log_prob(action)  # -> (1,)

        obs, reward, terminated, truncated, _ = env.step(action.item())

        action_list.append(action.item())
        log_prob_list.append(log_prob.cpu())
        reward_list.append(float(reward))
        value_list.append(value.cpu())

    # V(s_T): 0 if truly terminal, critic estimate if truncated
    if truncated and not terminated:
        obs_t = rearrange(torch.from_numpy(np.array(obs)), "frames h w -> 1 frames h w").to(device)
        with torch.no_grad():
            _, last_val = policy(obs_t)
        last_value = last_val.item()
    else:
        last_value = 0.0

    observations = torch.stack([torch.from_numpy(o) for o in obs_list])  # (T, frames, H, W)
    actions = torch.tensor(action_list, dtype=torch.long)  # (T,)
    old_log_probs = rearrange(torch.stack(log_prob_list), "t 1 -> t")  # (T,)
    values = rearrange(torch.stack(value_list), "t 1 -> t")  # (T,)

    return RolloutData(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        rewards=reward_list,
        values=values,
        last_value=last_value,
        episode_return=sum(reward_list),
        length=len(reward_list),
    )


def compute_gae(
    rewards: list[float],
    values: torch.Tensor,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Returns (advantages, returns) both as (T,) tensors.
    returns = advantages + values, used as value function targets.
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1].item()
        delta = rewards[t] + gamma * next_value - values[t].item()
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values.detach()
    return advantages, returns


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_epsilon: float,
    value_loss_coeff: float,
    entropy_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO clipped surrogate objective with value loss and entropy bonus.

    All inputs are (MB,) tensors (minibatch).
    Returns (total_loss, policy_loss, value_loss, entropy) as scalars.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)  # (MB,)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    value_loss = F.mse_loss(new_values, returns)

    entropy_val = entropy.mean()

    total_loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy_val
    return total_loss, policy_loss, value_loss, entropy_val


# ─── Training loop ──────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    episode: int
    episode_return: float
    running_reward: float
    episode_length: int
    loss: float
    policy_loss: float
    value_loss: float
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

    policy = ActorCriticNetwork(
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
    t_start = time.time()

    for episode in range(config.training.max_episodes):
        rollout = collect_rollout(env, policy, device)

        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.last_value,
            config.training.gamma,
            config.training.gae_lambda,
        )

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()

        obs_device = rollout.observations.to(device)  # (T, frames, H, W)
        actions_device = rollout.actions.to(device)  # (T,)
        old_lp_device = rollout.old_log_probs.to(device)  # (T,)
        adv_device = advantages.to(device)  # (T,)
        ret_device = returns.to(device)  # (T,)

        T = rollout.length
        epoch_total: list[float] = []
        epoch_policy: list[float] = []
        epoch_value: list[float] = []
        epoch_entropy: list[float] = []

        for _ppo_epoch in range(config.training.ppo_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, config.training.minibatch_size):
                mb_idx = indices[start : start + config.training.minibatch_size]

                mb_obs = obs_device[mb_idx]  # (MB, frames, H, W)
                mb_actions = actions_device[mb_idx]  # (MB,)
                mb_old_lp = old_lp_device[mb_idx]  # (MB,)
                mb_adv = adv_device[mb_idx]  # (MB,)
                mb_ret = ret_device[mb_idx]  # (MB,)

                dist, new_values = policy(mb_obs)
                new_log_probs = dist.log_prob(mb_actions)  # (MB,)
                entropy = dist.entropy()  # (MB,)

                total, p_loss, v_loss, ent = ppo_loss(
                    old_log_probs=mb_old_lp,
                    new_log_probs=new_log_probs,
                    advantages=mb_adv,
                    new_values=new_values,
                    returns=mb_ret,
                    entropy=entropy,
                    clip_epsilon=config.training.clip_epsilon,
                    value_loss_coeff=config.training.value_loss_coeff,
                    entropy_coeff=config.training.entropy_coeff,
                )

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                epoch_total.append(total.item())
                epoch_policy.append(p_loss.item())
                epoch_value.append(v_loss.item())
                epoch_entropy.append(ent.item())

        running_reward = (
            rollout.episode_return if episode == 0 else 0.05 * rollout.episode_return + 0.95 * running_reward
        )

        avg_total = sum(epoch_total) / len(epoch_total)
        avg_policy = sum(epoch_policy) / len(epoch_policy)
        avg_value = sum(epoch_value) / len(epoch_value)
        avg_entropy = sum(epoch_entropy) / len(epoch_entropy)

        metrics = StepMetrics(
            episode=episode,
            episode_return=rollout.episode_return,
            running_reward=running_reward,
            episode_length=rollout.length,
            loss=avg_total,
            policy_loss=avg_policy,
            value_loss=avg_value,
            entropy=avg_entropy,
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
                f"p_loss={metrics.policy_loss:>8.4f}  "
                f"v_loss={metrics.value_loss:>8.4f}  "
                f"entropy={metrics.entropy:>6.4f}  "
                f"t={metrics.elapsed_sec:>6.1f}s"
            )
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
            torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "episode": episode}, path)
            print(f"  → saved {path}")

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

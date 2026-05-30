"""Doom RL training: actor-critic network (CNN or LSTM), PPO with GAE, Hydra entrypoint.

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

# Type alias for LSTM hidden state: (h, c) each of shape (num_layers, batch, hidden_dim)
LSTMHidden = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "cnn"
    cnn_channels: tuple[int, ...] = (32, 64, 64)
    hidden_dim: int = 512
    lstm_layers: int = 1


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
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class Paths:
    root_working_dir: str = "/tmp"
    model_name: str = "default"


@dataclass(frozen=True)
class Config:
    env: EnvConfig = None
    model: ModelConfig = None
    training: TrainingHparams = None
    paths: Paths = None


def build_config(cfg: DictConfig) -> Config:
    return Config(
        env=EnvConfig(**cfg.env),
        model=ModelConfig(
            arch=cfg.model.arch,
            cnn_channels=tuple(cfg.model.cnn_channels),
            hidden_dim=cfg.model.hidden_dim,
            lstm_layers=cfg.model.lstm_layers,
        ),
        training=TrainingHparams(**cfg.training),
        paths=Paths(**cfg.paths),
    )


# ─── Network ────────────────────────────────────────────────────────────────


def _build_conv_layers(in_channels: int, cnn_channels: tuple[int, ...]) -> nn.ModuleList:
    """Build Nature DQN-style conv layers from channel spec.

    Uses fixed kernel/stride pattern: 8x4, 4x2, 3x1 (matching Nature DQN).
    """
    kernel_strides = [(8, 4), (4, 2), (3, 1)]
    layers = nn.ModuleList()
    ch_in = in_channels
    for ch_out, (kernel, stride) in zip(cnn_channels, kernel_strides):
        layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=kernel, stride=stride))
        ch_in = ch_out
    return layers


def _conv_forward(layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        x = F.relu(layer(x))
    return x


def _compute_conv_output_size(conv_layers: nn.ModuleList, in_channels: int, resolution: int) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, resolution, resolution)
        out = _conv_forward(conv_layers, dummy)
        return out.numel()


class ActorCriticNetwork(nn.Module):
    """Nature DQN CNN backbone + policy and value heads.

    Input: (B, frame_stack, H, W) float32 in [0, 1]
    Output: (Categorical distribution over actions, value estimates)
    """

    def __init__(self, model_config: ModelConfig, frame_stack: int, num_actions: int, resolution: int = 84):
        super().__init__()
        self.conv_layers = _build_conv_layers(frame_stack, model_config.cnn_channels)
        conv_flat = _compute_conv_output_size(self.conv_layers, frame_stack, resolution)

        self.fc = nn.Linear(conv_flat, model_config.hidden_dim)
        self.policy_head = nn.Linear(model_config.hidden_dim, num_actions)
        self.value_head = nn.Linear(model_config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        # x: (batch, frames, h, w)
        x = _conv_forward(self.conv_layers, x)  # -> (batch, channels, h', w')
        x = rearrange(x, "batch channels h w -> batch (channels h w)")
        x = F.relu(self.fc(x))  # -> (batch, hidden)
        logits = self.policy_head(x)  # -> (batch, actions)
        value = self.value_head(x)  # -> (batch, 1)
        value = rearrange(value, "batch 1 -> batch")
        return torch.distributions.Categorical(logits=logits), value


class LSTMActorCriticNetwork(nn.Module):
    """CNN backbone + LSTM + policy and value heads.

    Input: (B, frame_stack, H, W) float32 in [0, 1], plus LSTM hidden state
    Output: (Categorical distribution, value estimate, new hidden state)
    """

    def __init__(self, model_config: ModelConfig, frame_stack: int, num_actions: int, resolution: int = 84):
        super().__init__()
        self.hidden_dim = model_config.hidden_dim
        self.num_layers = model_config.lstm_layers

        self.conv_layers = _build_conv_layers(frame_stack, model_config.cnn_channels)
        conv_flat = _compute_conv_output_size(self.conv_layers, frame_stack, resolution)

        self.lstm = nn.LSTM(
            input_size=conv_flat,
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.lstm_layers,
            batch_first=True,
        )
        self.policy_head = nn.Linear(model_config.hidden_dim, num_actions)
        self.value_head = nn.Linear(model_config.hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> LSTMHidden:
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self, x: torch.Tensor, hidden: LSTMHidden
    ) -> tuple[torch.distributions.Categorical, torch.Tensor, LSTMHidden]:
        # x: (batch, frames, h, w) — single timestep
        x = _conv_forward(self.conv_layers, x)  # -> (batch, channels, h', w')
        x = rearrange(x, "batch channels h w -> batch (channels h w)")

        # LSTM expects (batch, seq_len, features) with batch_first=True
        x = rearrange(x, "batch features -> batch 1 features")
        lstm_out, new_hidden = self.lstm(x, hidden)  # -> (batch, 1, hidden_dim)
        x = rearrange(lstm_out, "batch 1 hidden -> batch hidden")

        logits = self.policy_head(x)  # -> (batch, actions)
        value = self.value_head(x)  # -> (batch, 1)
        value = rearrange(value, "batch 1 -> batch")
        return torch.distributions.Categorical(logits=logits), value, new_hidden

    def forward_sequence(
        self, x: torch.Tensor, hidden: LSTMHidden
    ) -> tuple[torch.distributions.Categorical, torch.Tensor, LSTMHidden]:
        """Forward pass over a full sequence for PPO updates.

        x: (batch, seq_len, frames, h, w)
        Returns distributions and values for all timesteps at once.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Process all frames through CNN in one batch
        x = rearrange(x, "batch seq frames h w -> (batch seq) frames h w")
        x = _conv_forward(self.conv_layers, x)  # -> (batch*seq, channels, h', w')
        x = rearrange(x, "bs channels h w -> bs (channels h w)")
        x = rearrange(x, "(batch seq) features -> batch seq features", batch=batch_size, seq=seq_len)

        lstm_out, new_hidden = self.lstm(x, hidden)  # -> (batch, seq, hidden_dim)
        lstm_out = rearrange(lstm_out, "batch seq hidden -> (batch seq) hidden")

        logits = self.policy_head(lstm_out)  # -> (batch*seq, actions)
        value = self.value_head(lstm_out)  # -> (batch*seq, 1)

        logits = rearrange(logits, "(batch seq) actions -> batch seq actions", batch=batch_size, seq=seq_len)
        value = rearrange(value, "(batch seq) 1 -> batch seq", batch=batch_size, seq=seq_len)

        # Squeeze batch dim since we always have batch=1 for episode processing
        logits = rearrange(logits, "1 seq actions -> seq actions")
        value = rearrange(value, "1 seq -> seq")

        return torch.distributions.Categorical(logits=logits), value, new_hidden


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
    """Roll out one full episode with CNN policy, collecting everything PPO needs."""
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


def collect_rollout_lstm(env: gymnasium.Env, policy: LSTMActorCriticNetwork, device: torch.device) -> RolloutData:
    """Roll out one full episode with LSTM policy, threading hidden state through steps."""
    obs, _ = env.reset()
    hidden = policy.init_hidden(batch_size=1, device=device)

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
            dist, value, hidden = policy(obs_t, hidden)
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
            _, last_val, _ = policy(obs_t, hidden)
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


def _ppo_update_cnn(
    policy: ActorCriticNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutData,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: Config,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """PPO update with shuffled minibatches for CNN policy."""
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
            nn.utils.clip_grad_norm_(policy.parameters(), config.training.max_grad_norm)
            optimizer.step()

            epoch_total.append(total.item())
            epoch_policy.append(p_loss.item())
            epoch_value.append(v_loss.item())
            epoch_entropy.append(ent.item())

    return (
        sum(epoch_total) / len(epoch_total),
        sum(epoch_policy) / len(epoch_policy),
        sum(epoch_value) / len(epoch_value),
        sum(epoch_entropy) / len(epoch_entropy),
    )


def _ppo_update_lstm(
    policy: LSTMActorCriticNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutData,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: Config,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """PPO update for LSTM policy — processes full episode in order to preserve temporal dependencies."""
    obs_device = rollout.observations.to(device)  # (T, frames, H, W)
    actions_device = rollout.actions.to(device)  # (T,)
    old_lp_device = rollout.old_log_probs.to(device)  # (T,)
    adv_device = advantages.to(device)  # (T,)
    ret_device = returns.to(device)  # (T,)

    # Add batch dim: (1, T, frames, H, W)
    obs_seq = rearrange(obs_device, "t frames h w -> 1 t frames h w")

    epoch_total: list[float] = []
    epoch_policy: list[float] = []
    epoch_value: list[float] = []
    epoch_entropy: list[float] = []

    for _ppo_epoch in range(config.training.ppo_epochs):
        # Each epoch re-processes the full sequence from zero hidden state
        hidden = policy.init_hidden(batch_size=1, device=device)
        dist, new_values, _ = policy.forward_sequence(obs_seq, hidden)
        # dist covers all T steps, new_values: (T,)

        new_log_probs = dist.log_prob(actions_device)  # (T,)
        entropy = dist.entropy()  # (T,)

        total, p_loss, v_loss, ent = ppo_loss(
            old_log_probs=old_lp_device,
            new_log_probs=new_log_probs,
            advantages=adv_device,
            new_values=new_values,
            returns=ret_device,
            entropy=entropy,
            clip_epsilon=config.training.clip_epsilon,
            value_loss_coeff=config.training.value_loss_coeff,
            entropy_coeff=config.training.entropy_coeff,
        )

        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), config.training.max_grad_norm)
        optimizer.step()

        epoch_total.append(total.item())
        epoch_policy.append(p_loss.item())
        epoch_value.append(v_loss.item())
        epoch_entropy.append(ent.item())

    return (
        sum(epoch_total) / len(epoch_total),
        sum(epoch_policy) / len(epoch_policy),
        sum(epoch_value) / len(epoch_value),
        sum(epoch_entropy) / len(epoch_entropy),
    )


def train(config: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.training.seed)

    env = make_env(config.env, seed=config.training.seed)
    num_actions = env.action_space.n

    use_lstm = config.model.arch == "lstm"

    if use_lstm:
        policy: nn.Module = LSTMActorCriticNetwork(
            model_config=config.model,
            frame_stack=config.env.frame_stack,
            num_actions=num_actions,
            resolution=config.env.resolution,
        ).to(device)
    else:
        policy = ActorCriticNetwork(
            model_config=config.model,
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
        if use_lstm:
            rollout = collect_rollout_lstm(env, policy, device)
        else:
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
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()

        if use_lstm:
            avg_total, avg_policy, avg_value, avg_entropy = _ppo_update_lstm(
                policy, optimizer, rollout, advantages, returns, config, device
            )
        else:
            avg_total, avg_policy, avg_value, avg_entropy = _ppo_update_cnn(
                policy, optimizer, rollout, advantages, returns, config, device
            )

        running_reward = (
            rollout.episode_return if episode == 0 else 0.05 * rollout.episode_return + 0.95 * running_reward
        )

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

    final_path = os.path.join(model_dir, "checkpoint_final.pt")
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "episode": episode}, final_path)
    print(f"  → saved {final_path}")

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

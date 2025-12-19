from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class HParams:
    num_actions: int
    num_envs: int = 32
    num_steps: int = 2048
    # optimisation
    budget: int = 1_000_000
    num_epochs: int = 1
    num_minibatches: int = 8
    minibatch_size: int | None = None  # if None, derived from num_minibatches
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    max_episode_length: int = 20
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    lr: float = 1e-3
    normalise_advantage: bool = True
    clip_value_loss: bool = True
    gae_lambda: float = 0.95
    discount: float = 0.99
    weight_decay: float = 0.01
    anneal_lr: bool = True
    lambda_kld: float = 0.0  # KL divergence penalty coefficient (0 = disabled)
    # logs
    log_frequency: int = 1
    log_to_wandb: bool = False
    wandb_project_name: str = "interdiff"
    save_dir: str = field(default_factory=str)
    eval_frequency: int = 1
    random_start: bool = False


@dataclass
class Buffer:
    obs: torch.Tensor
    """(E, T, *obs_shape) pre-action observations"""
    action: torch.Tensor
    """(E, T) long"""
    log_prob: torch.Tensor
    """(E, T) old log-probs"""
    value: torch.Tensor
    """(E, T+1) bootstrap value at [:, -1]"""
    done: torch.Tensor
    """(E, T) bool"""
    reward: torch.Tensor
    """(E, T) float"""
    time: torch.Tensor
    """(E, T) (optional metadata)"""
    info: Dict[str, torch.Tensor]

    @classmethod
    def empty(
        cls,
        num_envs: int,
        num_steps: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
    ) -> "Buffer":
        E, T = num_envs, num_steps
        return cls(
            obs=torch.empty((E, T) + obs_shape, dtype=torch.float32, device=device),
            action=torch.empty((E, T), dtype=torch.long, device=device),
            log_prob=torch.empty((E, T), dtype=torch.float32, device=device),
            value=torch.empty((E, T + 1), dtype=torch.float32, device=device),
            done=torch.empty((E, T), dtype=torch.bool, device=device),
            reward=torch.empty((E, T), dtype=torch.float32, device=device),
            time=torch.empty((E, T), dtype=torch.float32, device=device),
            info={},
        )


class PPO(nn.Module):
    """Proximal Policy Optimization agent for language model finetuning.
    
    This PPO implementation is designed to work with GPT-like models that have:
    - An `encoder` attribute (TransformerEncoder)
    - An `lm_head` attribute (Linear layer for token prediction)
    
    The model's lm_head is used as the policy head (action distribution).
    A separate value head is created for value function estimation.
    
    Args:
        model: A GPT-like model with `encoder` and `lm_head` attributes.
               The lm_head output dimension determines the action space.
        optimiser: Optimizer for training. Can be None and set later.
        hparams: PPO hyperparameters.
        reference_model: Optional pretrained model for KL divergence regularization.
                        If provided, policy updates are penalized for diverging from this model.
    """
    def __init__(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer | None,
        hparams: HParams,
        reference_model: nn.Module | None = None,
    ):
        super().__init__()
        # Store the full model - we use its encoder and lm_head
        self.model = model
        
        # Store reference model for KL divergence (frozen, no gradients)
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        
        # The policy head is the model's lm_head (predicts next token = action)
        # We don't create a new one, we reuse the pretrained lm_head
        self.policy_head = model.lm_head
        
        # Create value head with same input dimension as lm_head
        # lm_head: (n_embd) -> (lm_head_out_size)
        # value_head: (n_embd) -> (1)
        n_embd = model.config.n_embd
        self.value_head = nn.Linear(n_embd, 1)
        
        self.optimiser = optimiser
        self.hparams = hparams
        self.env_steps = 0
        self.update_steps = 0
    
    @property
    def encoder(self) -> nn.Module:
        """Access the model's encoder."""
        return self.model.encoder

    def _encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation and return the embedding at the last valid position.
        
        Args:
            observation: Token IDs of shape (batch, seq_len).
            
        Returns:
            Embeddings of shape (batch, n_embd) at the last position.
        """
        # Convert to long for embedding lookup if needed
        if observation.dtype != torch.long:
            observation = observation.long()
        
        # Get encoder hidden states: (batch, seq_len, n_embd)
        hidden_states = self.model.encoder(observation)
        
        # Get embedding at the last position (where we predict the next action)
        # For autoregressive generation, we predict from the last token
        z = hidden_states[:, -1, :]  # (batch, n_embd)
        
        return z

    def policy(self, observation: torch.Tensor) -> torch.distributions.Categorical:
        z = self._encode(observation)
        logits = self.policy_head(z)
        return torch.distributions.Categorical(logits=logits)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        z = self._encode(observation)
        return self.value_head(z)

    def forward(self, observation: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Forward pass returning action distribution and value estimate.
        
        Args:
            observation: Token IDs of shape (batch, seq_len).
            
        Returns:
            Tuple of (Categorical distribution over actions, value estimates).
        """
        z = self._encode(observation)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)  # (batch,)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def collect_experience(self, env, timestep) -> Buffer:
        """
        Collect a batch of size E*T.
        Assumes timestep.observation is shape (E, *obs_shape).
        """
        obs0 = timestep.observation
        device = obs0.device if isinstance(obs0, torch.Tensor) else torch.device("cpu")
        obs_shape = tuple(obs0.shape[1:])  # drop E

        buffer = Buffer.empty(
            self.hparams.num_envs, self.hparams.num_steps, obs_shape, device
        )
        E, T = self.hparams.num_envs, self.hparams.num_steps

        # for episodic logging
        episodic_return = torch.zeros(E, device=device)
        episodic_len = torch.zeros(E, device=device)
        returns_out = torch.zeros(E, T, device=device)  # filled only at done steps
        ep_len_out = torch.zeros(E, T, device=device)
        for t in range(T):
            obs_t = timestep.observation # (E, *obs_shape)
            dist, value = self.forward(obs_t)
            action = dist.sample()  # (E,)
            logp = dist.log_prob(action)  # (E,)

            # step env
            next_ts = env.step(timestep, action)

            # write pre-action obs and on-policy stats
            buffer.obs[:, t] = obs_t
            buffer.action[:, t] = action
            buffer.log_prob[:, t] = logp
            buffer.value[:, t] = value
            buffer.reward[:, t] = next_ts.reward
            buffer.done[:, t] = next_ts.terminated
            buffer.time[:, t] = next_ts.t

            episodic_return += next_ts.reward
            episodic_len += 1

            # write episodic summaries at terminals
            if next_ts.terminated.any():
                d = next_ts.terminated
                returns_out[d, t] = episodic_return[d]
                ep_len_out[d, t] = episodic_len[d]
                episodic_return[d] = 0.0
                episodic_len[d] = 0.0
            timestep = next_ts


        # bootstrap value for the last observation
        
        last_value = self.value(timestep.observation).squeeze(-1)
        buffer.value[:, -1] = last_value

        # attach logging info tensors shaped like done
        buffer.info["returns"] = returns_out
        buffer.info["episode_lengths"] = ep_len_out
        return buffer

    def evaluate_experience(
        self, buf: Buffer
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (values_t, advantages, targets) with shapes (E,T)."""
        E, T = buf.reward.shape
        gamma, lam = self.hparams.discount, self.hparams.gae_lambda

        rewards = buf.reward
        dones = buf.done.float()
        values = buf.value  # (E, T+1)

        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(E, device=rewards.device)
        for t in reversed(range(T)):
            not_done = 1.0 - dones[:, t]
            delta = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]
            last_adv = delta + gamma * lam * not_done * last_adv
            advantages[:, t] = last_adv

        targets = advantages + values[:, :-1]  # V-targets
        if self.hparams.normalise_advantage:
            mean = advantages.mean()
            std = advantages.std(unbiased=False) + 1e-8
            advantages = (advantages - mean) / std

        return values[:, :-1], advantages, targets

    def update(self, env, timestep) -> Dict[str, Any]:
        buf = self.collect_experience(env, timestep)

        logs: Dict[str, float] = {
            "ppo/value_loss": 0.0,
            "ppo/policy_loss": 0.0,
            "ppo/entropy": 0.0,
            "ppo/loss": 0.0,
            "ppo/approx_kl": 0.0,
            "ppo/clip_frac": 0.0,
            "ppo/explained_variance": 0.0,
            "ppo/kl_div": 0.0,
        }

        E, T = self.hparams.num_envs, self.hparams.num_steps
        bsize = E * T
        obs_shape = buf.obs.shape[2:]

        # flatten helpers
        def flat_obs(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(bsize, *obs_shape)

        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(bsize)

        # choose minibatch size
        mb_size = (
            self.hparams.minibatch_size
            if self.hparams.minibatch_size is not None
            else max(1, bsize // self.hparams.num_minibatches)
        )

        num_updates = 0
        for _ in range(self.hparams.num_epochs):
            values, advantages, targets = self.evaluate_experience(buf)

            idx_chunks = torch.randperm(bsize, device=buf.obs.device).split(mb_size)
            for idx in idx_chunks:
                b_obs = flat_obs(buf.obs)[idx]
                b_actions = flat(buf.action)[idx]
                b_old_logp = flat(buf.log_prob)[idx]
                b_adv = flat(advantages)[idx]
                b_targ = flat(targets)[idx]
                b_old_v = flat(values)[idx]  # for value clipping baseline

                dist, value = self.forward(b_obs)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()  # tensor, keeps grad

                # policy loss with clipping
                ratio = (new_logp - b_old_logp).exp()
                surr1 = ratio * b_adv
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.hparams.clip_eps, 1.0 + self.hparams.clip_eps
                    )
                    * b_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # KL divergence with reference model (if enabled)
                kl_div = torch.tensor(0.0, device=b_obs.device)
                if self.reference_model is not None and self.hparams.lambda_kld > 0:
                    with torch.no_grad():
                        # Get reference model's policy distribution
                        # Ensure observations are long type for embedding layer
                        b_obs_long = b_obs.long() if b_obs.dtype != torch.long else b_obs
                        ref_z = self.reference_model.encoder(b_obs_long)[:, -1, :]
                        ref_logits = self.reference_model.lm_head(ref_z)
                        ref_dist = Categorical(logits=ref_logits)
                    
                    # KL divergence: KL(current || reference)
                    kl_div = torch.distributions.kl_divergence(dist, ref_dist).mean()

                # value loss (clipped or unclipped)
                v_pred = value  # (mb,)
                if self.hparams.clip_value_loss:
                    v_clipped = b_old_v + torch.clamp(
                        v_pred - b_old_v, -self.hparams.clip_eps, self.hparams.clip_eps
                    )
                    v_loss_unclipped = (v_pred - b_targ).pow(2)
                    v_loss_clipped = (v_clipped - b_targ).pow(2)
                    value_loss = (
                        0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (v_pred - b_targ).pow(2).mean()

                loss = (
                    self.hparams.vf_coef * value_loss
                    + policy_loss
                    - self.hparams.ent_coef * entropy
                    + self.hparams.lambda_kld * kl_div
                )

                self.optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if (
                    self.hparams.max_grad_norm is not None
                    and self.hparams.max_grad_norm > 0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.hparams.max_grad_norm
                    )
                self.optimiser.step()

                # anneal LR
                if self.hparams.anneal_lr:
                    frac = max(0.0, 1.0 - (self.update_steps / self.hparams.budget))
                    lr_now = float(self.hparams.lr) * frac
                    for pg in self.optimiser.param_groups:
                        pg["lr"] = lr_now

                # metrics
                approx_kl = 0.5 * (new_logp - b_old_logp).pow(2).mean()
                clip_frac = (
                    (ratio.sub(1.0).abs() > self.hparams.clip_eps).float().mean()
                )

                # explained variance: 1 - Var(y - yhat) / Var(y)
                ev = 1.0 - torch.var(b_targ - v_pred) / (torch.var(b_targ) + 1e-8)

                # accumulate (detach for logging)
                logs["ppo/value_loss"] += value_loss.detach().item()
                logs["ppo/policy_loss"] += policy_loss.detach().item()
                logs["ppo/entropy"] += entropy.detach().item()
                logs["ppo/loss"] += loss.detach().item()
                logs["ppo/approx_kl"] += approx_kl.detach().item()
                logs["ppo/clip_frac"] += clip_frac.detach().item()
                logs["ppo/explained_variance"] += ev.detach().item()
                logs["ppo/kl_div"] += kl_div.detach().item()

                self.update_steps += 1
                num_updates += 1

        # averages
        for k in list(logs.keys()):
            logs[k] /= max(1, num_updates)

        self.env_steps += E * T

        logs["rl/update_steps"] = self.update_steps
        logs["rl/env_steps"] = self.env_steps

        # episode stats from batch terminals
        mask = buf.done  # (E,T) bool
        term_returns = buf.info["returns"][mask]
        term_lengths = buf.info["episode_lengths"][mask]
        if term_returns.numel() > 0:
            logs["rl/returns"] = term_returns.mean().item()
            logs["rl/episode_length"] = term_lengths.mean().item()
        else:
            logs["rl/returns"] = 0.0
            logs["rl/episode_length"] = 0.0

        return logs

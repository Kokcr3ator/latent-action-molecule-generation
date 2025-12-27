from __future__ import annotations

from typing import Dict
from dataclasses import dataclass, field, replace

from tokenizers import Tokenizer
import torch

from .models import DynamicsModel, LatentActionModel
from .metrics import qed, logp, molecular_weight, tpsa, synthetic_accessibility
from .utils.eval_utils import tokens_to_smiles
from interdiff.utils.eval_utils import sample_from_logits

@dataclass
class Timestep:
    """Container for one environment transition."""

    observation: torch.Tensor
    action_observation: torch.Tensor
    t: torch.Tensor
    reward: torch.Tensor
    step_type: torch.Tensor
    rng: torch.Tensor
    info: Dict[str, torch.Tensor]

    @property
    def terminated(self) -> torch.Tensor:
        """Returns a boolean tensor indicating which environments have terminated."""
        return self.step_type > 0
    
    def replace(self, **kwargs) -> Timestep:
        return replace(self, **kwargs)

@dataclass
class DiscreteSpace:
    lower: int
    upper: int
    dtype: torch.dtype

@dataclass(kw_only=True)
class Reward:
    """Reward function for the environment."""

    eos_token_id: int
    task: str
    tokeniser: Tokenizer
    device: str

    def reward_from_smiles(self, smiles: str) -> float:
        if self.task == "qed":
            return qed(smiles, as_reward=True)
        elif self.task == "logp":
            return logp(smiles, as_reward=True)
        elif self.task == "mw":
            return molecular_weight(smiles, as_reward=True)
        elif self.task == "tpsa":
            return tpsa(smiles, as_reward=True)
        elif self.task == "sa":
            return synthetic_accessibility(smiles, as_reward=True)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
    def __call__(self, timestep: Timestep, next_obs: torch.Tensor) -> torch.Tensor:
        """Calculate the reward for the given action."""
        # if there is no termination (action != eos_token_id), reward is 0
        next_state = next_obs[torch.arange(next_obs.shape[0]), timestep.t + 1]
        if torch.all(next_state != self.eos_token_id):
            return torch.zeros_like(timestep.reward).to(self.device)

        smiles = tokens_to_smiles(token_ids=timestep.observation, tokenizer=self.tokeniser)
        new_reward = torch.as_tensor(
            [
                self.reward_from_smiles(smi)
                for smi in smiles
            ],
            dtype=torch.float32,
        ).to(self.device)
        not_ended = next_state != self.eos_token_id
        new_reward[not_ended] = 0.0
        return new_reward


@dataclass(kw_only=True)
class Env:
    action_space: DiscreteSpace = field(
        default_factory=lambda: DiscreteSpace(lower=0, upper=65535, dtype=torch.long)
    )
    """Action space for an individual environment (not vmapped)"""
    observation_space: DiscreteSpace = field(
        default_factory=lambda: DiscreteSpace(lower=0, upper=65535, dtype=torch.long)
    )
    """Observation space for an individual environment (not vmapped)"""
    num_envs: int = 1
    """Number of environments to simulate in parallel"""
    context_length: int = 256
    """Defines the length of the observation. Unset items have value paspecial_tokens["pad"]"""
    max_steps: int = 100
    """Number of timesteps after which the environment times out and resets"""
    discount: float = 1.0
    """Discount factor for future rewards"""
    reward_fn: Reward = field(
        default_factory=lambda: Reward(
            eos_token_id=4,
            task="qed",
            tokeniser=None,
        )
    )
    """Reward function for the environment"""
    special_tokens: Dict[str, int] = field(
        default_factory=lambda: {
            "bos": 3,
            "eos": 4,
            "pad": 0,
        }
    )
    """Special tokens used in the environment, such as bos (beginning of sentence),
    eos (end of sentence), and pad (padding)"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    """Device to run the environment on"""
    random_start: bool = False
    """If true, the environment starts with only a bos token, otherwise it starts with random tokens"""
    def _initial_condition(self, seed):
        n_envs_to_reset = seed.shape[0]
        new_time = torch.zeros((n_envs_to_reset,), dtype=torch.long).to(self.device)
        # start with all pad tokens
        new_obs = torch.full(
            (n_envs_to_reset, self.context_length),
            self.special_tokens["pad"],
            dtype=torch.long,
        ).to(self.device)

        new_action_obs = torch.full(
            (n_envs_to_reset, self.context_length),
            self.special_tokens["pad"],
            dtype=torch.long,
        ).to(self.device)

        # add bos token at the start
        new_obs[:, 0] = self.special_tokens["bos"]
        new_reward = torch.zeros((n_envs_to_reset,), dtype=torch.float32).to(self.device)
        new_step_type = torch.zeros((n_envs_to_reset,), dtype=torch.long).to(self.device)
        new_returns = torch.zeros((n_envs_to_reset,), dtype=torch.float32).to(self.device)
        if self.random_start:
            new_obs = torch.scatter(
                new_obs,
                0,
                torch.as_tensor([1, 2, 3] * n_envs_to_reset).view(n_envs_to_reset, 3),
                torch.randint(
                    low=self.action_space.lower + len(self.special_tokens),
                    high=self.action_space.upper,
                    size=(n_envs_to_reset, 3),
                    dtype=torch.long,
                ),
            )
            new_time += 3
        new_rng = seed
        return new_obs, new_action_obs, new_time, new_reward, new_step_type, new_returns, new_rng
    
    def reset(self, timestep: Timestep, seed: torch.Tensor) -> Timestep:
        new_obs, new_action_obs, new_time, new_reward, new_step_type, new_returns, new_rng = self._initial_condition(seed)

        if timestep is None:
            # initial reset
            return Timestep(
                observation=new_obs,
                action_observation=new_action_obs,
                t=new_time,
                reward=new_reward,
                step_type=new_step_type,
                rng=new_rng,
                info={"returns": new_returns},
            )
        
        # start time
        reset_mask = timestep.step_type != 0
        n_envs_to_reset = seed.shape[0]
        # the number of true values in reset_mask should equal n_envs_to_reset
        assert torch.sum(reset_mask).item() == n_envs_to_reset, \
            f"Number of envs to reset ({n_envs_to_reset}) does not match the number of done envs ({torch.sum(reset_mask).item()})"
        return Timestep(
            observation=self._update_fn(timestep.observation, new_obs, reset_mask),
            action_observation=self._update_fn(timestep.action_observation, new_action_obs, reset_mask),
            t=self._update_fn(timestep.t, new_time, reset_mask),
            reward=self._update_fn(timestep.reward, new_reward, reset_mask),
            step_type=self._update_fn(timestep.step_type, new_step_type, reset_mask),
            rng=self._update_fn(timestep.rng, seed.to(torch.long), reset_mask),
            info={"returns": self._update_fn(timestep.info["returns"], new_returns, reset_mask)},
        )

    def _get_next_state(self, timestep: Timestep, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step(self, timestep: Timestep, action: torch.Tensor) -> Timestep:
        # autoreset if necessary: 0 = transition, 1 = truncation, 2 = termination

        # Environments that have finished on the *previous* step
        reset_mask = timestep.step_type != 0  # shape: (num_envs,)
        
        # reset done envs
        if torch.any(reset_mask):
            n_envs_to_reset = torch.sum(reset_mask).item()
            seeds = torch.ones(n_envs_to_reset).to(self.device) * (timestep.rng[reset_mask] + 1)
            timestep = self.reset(timestep, seeds)

        next_obs = self._get_next_state(timestep, action)
        next_action_obs = timestep.action_observation.clone()
        next_action_obs[torch.arange(next_action_obs.shape[0]), timestep.t] = action
        reward = self._reward(timestep, next_obs)
        step_type = self._termination(timestep = timestep, next_obs = next_obs)
        return Timestep(
            observation=next_obs,
            action_observation=next_action_obs,
            t=timestep.t + 1,
            reward=reward,
            step_type=step_type,
            rng=timestep.rng,
            info={
                "returns": timestep.info["returns"] + reward * self.discount**timestep.t
            },
        )
    
    def _update_fn(self, x, y, reset_mask):
        x[reset_mask] = y
        return x

    def _reward(self, timestep: Timestep, next_obs: torch.Tensor) -> torch.Tensor:
        return self.reward_fn(timestep, next_obs)

    def _termination(self, timestep: Timestep, next_obs: torch.Tensor) -> torch.Tensor:
        """Returns:
        - 0 if the environment still runs
        - 1 if the environment has terminated
        - 2 if the environment has reached a maximum length
        """
        next_state = next_obs[torch.arange(next_obs.shape[0]), timestep.t + 1]
        does_timeout = (timestep.t + 1 >= self.max_steps).to(torch.long)
        does_terminate = (next_state == self.special_tokens["eos"]).to(torch.long)
        step_type = torch.zeros((self.num_envs,), dtype=torch.long).to(self.device)
        # replace timeouts
        step_type[does_timeout == 1] = 2
        step_type[does_terminate == 1] = 1
        return step_type


class MoleculeGenerationEnv(Env):

    def _get_next_state(self, timestep: Timestep, action: torch.Tensor) -> torch.Tensor:
        next_obs = timestep.observation.clone()
        next_obs[torch.arange(next_obs.shape[0]), timestep.t + 1] = action
        return next_obs


class ControllableMoleculeGenerationEnv(Env):
    dynamics_model: DynamicsModel
    lam: LatentActionModel

    def _initial_condition(self, seed):
        n_envs_to_reset = seed.shape[0]
        new_time = torch.zeros((n_envs_to_reset,), dtype=torch.long).to(self.device)
        # start with all pad tokens
        new_obs = torch.full(
            (n_envs_to_reset, self.context_length),
            self.special_tokens["pad"],
            dtype=torch.long,
        ).to(self.device)

        new_action_obs = torch.full(
            (n_envs_to_reset, self.context_length),
            self.special_tokens["pad"],
            dtype=torch.long,
        ).to(self.device)

        # add bos token at the start
        new_obs[:, 0] = self.special_tokens["bos"]
        new_reward = torch.zeros((n_envs_to_reset,), dtype=torch.float32).to(self.device)
        new_step_type = torch.zeros((n_envs_to_reset,), dtype=torch.long).to(self.device)
        new_returns = torch.zeros((n_envs_to_reset,), dtype=torch.float32).to(self.device)
        if self.random_start:
            new_obs = torch.scatter(
                new_obs,
                0,
                torch.as_tensor([1, 2, 3] * n_envs_to_reset).view(n_envs_to_reset, 3),
                torch.randint(
                    low=self.action_space.lower + len(self.special_tokens),
                    high=self.action_space.upper,
                    size=(n_envs_to_reset, 3),
                    dtype=torch.long,
                ),
            )
            new_time += 3
            # compute the corresponding action_obs using the latent action model
            states = new_obs[:, :4]
            with torch.no_grad():
                _, _, actions = self.lam.vq_encode(states)
            new_action_obs[:, :3] = actions
        new_rng = seed
        return new_obs, new_action_obs, new_time, new_reward, new_step_type, new_returns, new_rng
    
    def _get_next_state(self, timestep: Timestep, action: torch.Tensor) -> torch.Tensor:
        next_obs = timestep.observation.clone()
        all_actions = timestep.action_observation.clone()
        all_actions[torch.arange(all_actions.shape[0]), timestep.t] = action
        action_emb = self.lam.vq.codebook[all_actions]
        state_logits = self.dynamics_model(timestep.observation, action_emb)
        state_logits = state_logits[torch.arange(state_logits.shape[0]), timestep.t]
        new_states = sample_from_logits(tensor_logits=state_logits)
        next_obs[torch.arange(next_obs.shape[0]), timestep.t + 1] = new_states.squeeze(-1)
        return next_obs
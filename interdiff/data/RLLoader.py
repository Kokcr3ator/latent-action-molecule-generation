import logging
from pathlib import Path
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from tokenizers import Tokenizer

from interdiff.models import GPT, PolicyNetwork, ControllableGPT, DynamicsModel
from interdiff.ppo import PPO
from interdiff.envs import MoleculeGenerationEnv, ControllableMoleculeGenerationEnv, Reward, DiscreteSpace

class RLLoader(ABC):
    """Abstract base class for RL Loaders."""
    def __init__(self, cfg: DictConfig, device: torch.device, tokenizer: Tokenizer) -> None:
        self.cfg = cfg
        self.device = device
        self.tokenizer = tokenizer

    @abstractmethod
    def load_pretrained_model(self):
        """Load a pretrained model from checkpoint or instantiate from scratch."""
        pass
    @abstractmethod
    def setup_environment(self):
        """Setup the molecule generation environment with reward function."""
        pass
    def setup_ppo_agent(self, model, reference_model) -> PPO:
        """Create PPO agent with optimizer and reference model.
        
        The PPO agent uses the model's lm_head as the policy head
        and creates a separate value head. The reference model is used
        for KL divergence regularization.
        
        Args:
            model: The model to be trained (will be updated).
            reference_model: Frozen copy of pretrained model for KL divergence.
            cfg: Hydra configuration.
            device: Device to use.
            
        Returns:
            PPO agent with optimizer configured.
        """
        
        # Create HParams from config
        hparams = instantiate(self.cfg.ppo)
        # Create PPO agent
        ppo_agent = PPO(
            model=model,
            optimiser=None,  # Set after to include all parameters
            hparams=hparams,
            reference_model=reference_model,  # Add reference model for KL divergence
        )
        
        # Create optimizer with all PPO parameters (model + value_head)
        optimizer = torch.optim.AdamW(
            ppo_agent.parameters(),
            lr=self.cfg.ppo.lr,
            weight_decay=self.cfg.ppo.weight_decay,
            betas=(self.cfg.optim.beta1, self.cfg.optim.beta2),
        )
        ppo_agent.optimiser = optimizer

        return ppo_agent.to(self.device)

class FinetuneBaseLoader(RLLoader):
    def __init__(self, cfg: DictConfig, device: torch.device, tokenizer: Tokenizer) -> None:
        super().__init__(cfg, device, tokenizer)
    
    def load_pretrained_model(self) -> GPT:
        """Load a pretrained GPT model from checkpoint or instantiate from scratch."""

        init_from = self.cfg.ckpt.get("init_from", "scratch")

        if init_from == "resume":
            ckpt_dir = Path(self.cfg.ckpt.path)
            ckpt_path = ckpt_dir / self.cfg.ckpt.ckpt_name
            if ckpt_path.exists():
                logging.info(f"Loading pretrained model from {ckpt_path}")
                model = GPT.load(str(ckpt_path))
        else:
            logging.info("Initializing model from scratch (no pretraining)")
        
        return model.to(self.device)

    def setup_environment(self, model) -> MoleculeGenerationEnv:
        """Create the molecule generation environment with reward function.
        
        Sets the action and observation spaces based on model configuration:
        - action_space: [0, lm_head_out_size) - tokens the model can generate
        - observation_space: [0, vocab_size) - tokens in input sequences
        """
        
        # Create reward function
        reward_fn = Reward(
            eos_token_id=self.cfg.tokenizer.eos_token_id,
            task=self.cfg.reward.task,
            tokeniser=self.tokenizer,
            device=self.device,
        )
        
        # Create action and observation spaces from model config
        # action_space upper bound is lm_head output size (tokens model can generate)
        action_space = DiscreteSpace(
            lower=0,
            upper=model.lm_head_out_size,
            dtype=torch.long,
        )
        # observation_space upper bound is vocab size (tokens in input)
        observation_space = DiscreteSpace(
            lower=0,
            upper=model.vocab_size,
            dtype=torch.long,
        )
        
        # Create environment
        env = MoleculeGenerationEnv(
            num_envs=self.cfg.ppo.num_envs,
            context_length=self.cfg.context.seq_len,
            discount=self.cfg.ppo.discount,
            reward_fn=reward_fn,
            special_tokens={
                "bos": self.cfg.tokenizer.bos_token_id,
                "eos": self.cfg.tokenizer.eos_token_id,
                "pad": self.cfg.tokenizer.pad_token_id,
            },
            action_space=action_space,
            observation_space=observation_space,
            device=self.device,
            random_start=self.cfg.ppo.random_start,
        )
        
        return env

class FinetuneControlable(RLLoader):
    def __init__(self, cfg: DictConfig, device: torch.device, tokenizer: Tokenizer) -> None:
        super().__init__(cfg, device, tokenizer)

    def load_pretrained_model(self) -> PolicyNetwork:
        """Load a pretrained PolicyNetwork model from checkpoint or instantiate from scratch."""

        init_from = self.cfg.ckpt.get("init_from", "scratch")

        if init_from == "resume":
            ckpt_dir = Path(self.cfg.ckpt.path)
            ckpt_path = ckpt_dir / self.cfg.ckpt.ckpt_name
            if ckpt_path.exists():
                logging.info(f"Loading pretrained model from {ckpt_path}")
                model = PolicyNetwork.load(str(ckpt_path))
        else:
            logging.info("Initializing model from scratch (no pretraining)")
        
        return model.to(self.device)

    def load_dynamics_model(self) -> DynamicsModel:
        """Load a pretrained DynamicsModel from checkpoint."""
        ckpt_dir = Path(self.cfg.loader.ckpt_controllable_path)
        ckpt_path = ckpt_dir / self.cfg.loader.ckpt_name
        if ckpt_path.exists():
            logging.info(f"Loading ControllableGPT from {ckpt_path}")
            controllable_gpt = ControllableGPT.load(str(ckpt_path))
        else:
            raise FileNotFoundError(f"ControllableGPT model checkpoint not found at {ckpt_path}")

        return controllable_gpt.dynamics_model.to(self.device)


    def setup_environment(self, model) -> ControllableMoleculeGenerationEnv:
        """Create the molecule generation environment with reward function.
        
        Sets the action and observation spaces based on model configuration:
        - action_space: [0, lm_head_out_size) - tokens the model can generate
        - observation_space: [0, vocab_size) - tokens in input sequences
        """
        
        # Create reward function
        reward_fn = Reward(
            eos_token_id=self.cfg.tokenizer.eos_token_id,
            task=self.cfg.reward.task,
            tokeniser=self.tokenizer,
            device=self.device,
        )
        
        # Create action and observation spaces from model config
        # action_space upper bound is lm_head output size (tokens model can generate)
        action_space = DiscreteSpace(
            lower=0,
            upper=self.model.lm_head_out_size,
            dtype=torch.long,
        )
        # observation_space upper bound is vocab size (tokens in input)
        observation_space = DiscreteSpace(
            lower=0,
            upper=model.vocab_size,
            dtype=torch.long,
        )
        
        # Create environment
        env = ControllableMoleculeGenerationEnv(
            num_envs=self.cfg.ppo.num_envs,
            context_length=self.cfg.context.seq_len,
            discount=self.cfg.ppo.discount,
            reward_fn=reward_fn,
            special_tokens={
                "bos": self.cfg.tokenizer.bos_token_id,
                "eos": self.cfg.tokenizer.eos_token_id,
                "pad": self.cfg.tokenizer.pad_token_id,
            },
            action_space=action_space,
            observation_space=observation_space,
            device=self.device,
            random_start=self.cfg.ppo.random_start,
        )
        env.dynamics_model = self.load_dynamics_model()
        
        return env
"""PPO finetuning script for molecule generation.

This script finetunes a pretrained GPT model using Proximal Policy Optimization (PPO)
for molecule generation with a specified reward function (e.g., QED, LogP).

Usage:
    python scripts/train_ppo.py
    python scripts/train_ppo.py reward.task=logp ppo.num_envs=64
    python scripts/train_ppo.py ckpt.init_from=resume ckpt.path=path/to/pretrained
"""
import os
import logging
from copy import deepcopy

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch

from interdiff.utils.torch_utils import seed_all
from interdiff.trainers.base_RL import RLTrainerBase
from interdiff.io import load_tokenizer
from scripts.tokenise_dataset import run_tokenisation
from interdiff.data.GPTLoader import NextTokenDataset
from torch.utils.data import DataLoader

@hydra.main(version_base=None, config_path="../interdiff/conf", config_name="ppo_config")
def main(cfg: DictConfig):
    """Main PPO finetuning entry point."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("train_ppo")
    
    # Set random seed
    seed_all(cfg.seed)
    
    # Run tokenization to ensure dataset is ready
    save_path = run_tokenisation(cfg)
    log.info(f"Using tokenised dataset at {save_path}")
    
    # Setup device
    device = torch.device(cfg.sys.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer(os.path.join(cfg.rl_train_cfg.tokenizer_dir, "tokenizer.json"))
    log.info(f"Tokenizer loaded with vocab size {tokenizer.get_vocab_size()}")
    log.info("Setting up loader...")
    loader = instantiate(cfg.loader)
    log.info("Loading pretrained model...")
    model = loader.load_pretrained_model(cfg, device)
    log.info(f"Pretrained model loaded with embedding dim {model.config.n_embd}")
    
    # Create a reference copy of the pretrained model for KL divergence
    # This copy will remain frozen during training
    log.info("Creating reference model copy for KL divergence...")
    reference_model = None
    if cfg.ppo.lambda_kld > 0:
        reference_model = deepcopy(model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        log.info(f"Reference model created (frozen) with lambda_kld={cfg.ppo.lambda_kld}")
    else:
        log.info("KL divergence disabled (lambda_kld=0)")
    
    # Setup environment with action/observation spaces from model config
    log.info(f"Setting up environment with reward task: {cfg.reward.task}")
    env = loader.setup_environment(cfg, model, tokenizer, device)
    log.info(f"Environment created with {env.num_envs} parallel envs")
    log.info(f"  Action space: [0, {env.action_space.upper})")
    log.info(f"  Observation space: [0, {env.observation_space.upper})")
    
    # Setup PPO agent with the full model
    log.info("Setting up PPO agent...")
    ppo_agent = loader.setup_ppo_agent(cfg, model, reference_model, device)
    log.info("PPO agent created")
    
    # Setup logger
    logger = instantiate(cfg.log) if cfg.wandb_log else None
    
    # Create RL training config
    rl_train_cfg = instantiate(cfg.rl_train_cfg)
    
    # Load training dataloader for reference SMILES (novelty calculation)
    # Create a simple dataloader directly from the tokenized dataset
    log.info("Loading training dataloader for reference SMILES...")
    from safetensors import safe_open
    with safe_open(save_path, framework="pt") as f:
        tokens_tensor = f.get_tensor(list(f.keys())[0])
    
    # Ensure tokens are on CPU for the dataloader
    if tokens_tensor.is_cuda:
        tokens_tensor = tokens_tensor.cpu()
    
    dataset = NextTokenDataset(tokens_tensor, pad_token_id=cfg.tokenizer.pad_token_id)
    train_dataloader = DataLoader(
        dataset,
        batch_size=2048,  # Batch size for reference SMILES extraction
        shuffle=False,
        drop_last=False,
        pin_memory=False,  # Disable pin_memory to avoid CUDA pinning issues
    )
    log.info(f"Training dataloader created with {len(dataset)} samples")
    
    # Create trainer
    trainer = RLTrainerBase(
        ppo_agent=ppo_agent,
        env=env,
        logger=logger,
        train_cfg=rl_train_cfg,
    )
    
    # Log configuration
    log.info("=" * 60)
    log.info("PPO Finetuning Configuration:")
    log.info(f"  Reward task: {cfg.reward.task}")
    log.info(f"  Num envs: {cfg.ppo.num_envs}")
    log.info(f"  Steps per rollout: {cfg.ppo.num_steps}")
    log.info(f"  Budget: {cfg.ppo.budget} env steps")
    log.info(f"  Learning rate: {cfg.ppo.lr}")
    log.info(f"  KL divergence penalty (lambda_kld): {cfg.ppo.lambda_kld}")
    log.info(f"  Checkpoint path: {rl_train_cfg.ckpt_path}")
    log.info("=" * 60)
    
    # Run training with train_dataloader for novelty calculation
    trainer.fit(train_dataloader=train_dataloader)
    
    log.info("PPO finetuning completed!")


if __name__ == "__main__":
    main()

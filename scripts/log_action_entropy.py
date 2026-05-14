"""Compute the empirical conditional entropy H(A_t | S_{0:t}) and push it to
W&B as a horizontal reference line on the val/loss plot.

Two modes:

  gpt     Actions are the next tokens s_{t+1} — no model required.
          Gives the natural entropy of the SMILES/SELFIES sequences and is
          the lower bound for base GPT training.

  policy  Actions are VQ codebook indices produced by a specific LAM.  Pass
          the pre-generated actions safetensors file and the codebook size.

Usage:
    # GPT lower bound for all vocab sizes
    python -m scripts.log_action_entropy gpt \\
        --vocab-sizes 128 512 1024 2048 4096 \\
        --wandb.group exp7_gpt_vocab_scaling

    # Policy distillation lower bound
    python -m scripts.log_action_entropy policy \\
        --vocab-sizes 2048 \\
        --actions-path /path/to/actions_dataset_num_latent_actions_2048_vocab_size_2048.safetensors \\
        --num-latents  2048 \\
        --wandb.group  exp6_distillation_impact
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Annotated, List, Union

import torch
import tyro

from interdiff.dataset_entropy import empirical_conditional_entropy
from interdiff.io import _load_tensor_from_safetensors
from scripts.tokenise_dataset import run_tokenisation


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("log_action_entropy")


@dataclass
class WandbCfg:
    project: str = "interdiff"
    entity: str = "latent-action-interdiff"
    group: str = "action_entropy"
    dir: str = ""


@dataclass
class _Common:
    vocab_sizes: List[int]                  # one W&B run per vocab size
    data_smiles: str = "interdiff/data/zinc/zinc.txt"
    context_length: int = 128
    pad_token_id: int = 0
    max_steps: int = 20_000
    wandb: WandbCfg = field(default_factory=WandbCfg)


@dataclass
class GptCfg(_Common):
    """GPT mode: actions = next tokens s_{t+1}.  No model required."""


@dataclass
class PolicyCfg(_Common):
    """Policy mode: actions are VQ indices from a pre-generated actions file."""
    actions_path: str = ""
    num_latents: int = 2048


def _push_wandb(H: float, num_latents: int, vocab_size: int,
                max_steps: int, run_name: str, wandb_cfg: WandbCfg) -> None:
    import wandb
    run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group,
        name=run_name,
        dir=wandb_cfg.dir or None,
        config={"num_latents": num_latents, "conditional_entropy": H},
    )
    for step in range(0, max_steps + 1, 100):
        wandb.log({"val/loss": H, "train/loss": H}, step=step)
    run.summary["dataset/conditional_entropy"] = H
    run.summary["dataset/marginal_entropy"]    = math.log(num_latents)
    wandb.finish()


def main() -> None:
    cfg = tyro.cli(Union[
        Annotated[GptCfg,    tyro.conf.subcommand("gpt")],
        Annotated[PolicyCfg, tyro.conf.subcommand("policy")],
    ])

    for vocab_size in cfg.vocab_sizes:
        log.info(f"=== vocab_size={vocab_size} ===")

        _, dataset_path = run_tokenisation(
            data_smiles=cfg.data_smiles,
            vocab_size=vocab_size,
            context_length=cfg.context_length,
        )
        tokens = _load_tensor_from_safetensors(dataset_path).to(torch.long)
        log.info(f"Tokenised dataset: {tuple(tokens.shape)}")

        if isinstance(cfg, GptCfg):
            actions     = tokens[:, 1:].clone()
            num_latents = int(tokens.max().item()) + 1
            run_name    = f"entropy_lb_gpt_vocab{vocab_size}"
        else:
            if not cfg.actions_path:
                raise ValueError("--actions-path is required in policy mode")
            actions     = _load_tensor_from_safetensors(cfg.actions_path).to(torch.long)
            num_latents = cfg.num_latents
            run_name    = f"entropy_lb_policy_nlatents{num_latents}_vocab{vocab_size}"
            log.info(f"Loaded actions {tuple(actions.shape)}")

        log.info("Computing H(A | S_{0:t})...")
        H = empirical_conditional_entropy(
            tokens=tokens,
            actions=actions,
            num_latents=num_latents,
            pad_token_id=cfg.pad_token_id,
        )
        log.info(f"H(A | context) = {H:.4f} nats  (max = ln({num_latents}) = {math.log(num_latents):.4f})")

        log.info(f"Pushing to W&B run '{run_name}'...")
        _push_wandb(H=H, num_latents=num_latents, vocab_size=vocab_size,
                    max_steps=cfg.max_steps, run_name=run_name, wandb_cfg=cfg.wandb)

    log.info("Done.")


if __name__ == "__main__":
    main()

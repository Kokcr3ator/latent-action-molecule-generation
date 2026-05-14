"""Compute the empirical conditional entropy H(A_t | S_{0:t}) and push it to
W&B as a horizontal reference line on the val/loss plot.

Two modes:

  gpt     Actions are the next tokens s_{t+1} — no model needed, only the
          tokenised dataset.  This gives the natural entropy of the SMILES/
          SELFIES sequences and is the lower bound for base GPT training.

  policy  Actions are VQ codebook indices produced by a specific LAM.  Pass
          the pre-generated actions safetensors file and the codebook size.

Usage:
    # GPT lower bound — tokenised dataset only
    python -m scripts.log_action_entropy gpt \\
        --tokens-path /path/to/dataset.safetensors \\
        --wandb.group exp7_gpt_vocab_scaling

    # Policy distillation lower bound — pre-generated actions file
    python -m scripts.log_action_entropy policy \\
        --tokens-path  /path/to/dataset.safetensors \\
        --actions-path /path/to/actions_dataset_num_latent_actions_2048_vocab_size_2048.safetensors \\
        --num-latents  2048 \\
        --wandb.group  exp6_distillation_impact
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Annotated, Union

import torch
import tyro

from interdiff.dataset_entropy import empirical_conditional_entropy
from interdiff.io import _load_tensor_from_safetensors


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
    tokens_path: str                # safetensors [N, T] tokenised sequences
    pad_token_id: int = 0
    max_steps: int = 20_000
    wandb: WandbCfg = field(default_factory=WandbCfg)


@dataclass
class GptCfg(_Common):
    """GPT mode: actions = next tokens s_{t+1}.  No model required."""


@dataclass
class PolicyCfg(_Common):
    """Policy mode: actions are VQ indices from a pre-generated actions file."""
    actions_path: str = ""          # safetensors [N, T-1] action indices
    num_latents: int = 2048         # codebook size


def _run(tokens: torch.Tensor, actions: torch.Tensor, num_latents: int,
         pad_token_id: int, max_steps: int, run_name: str, wandb_cfg: WandbCfg) -> None:
    log.info("Computing H(A | S_{0:t})...")
    H = empirical_conditional_entropy(
        tokens=tokens,
        actions=actions,
        num_latents=num_latents,
        pad_token_id=pad_token_id,
    )
    H_max = math.log(num_latents)
    log.info(f"H(A | context) = {H:.4f} nats  (max = ln({num_latents}) = {H_max:.4f})")

    import wandb
    run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group,
        name=run_name,
        dir=wandb_cfg.dir or None,
        config={"num_latents": num_latents, "conditional_entropy": H},
    )

    log.info(f"Pushing horizontal line at {H:.4f} nats for steps 0..{max_steps}...")
    for step in range(0, max_steps + 1, 100):
        wandb.log({"val/loss": H}, step=step)

    run.summary["dataset/conditional_entropy"] = H
    run.summary["dataset/marginal_entropy"]    = H_max
    wandb.finish()
    log.info("Done.")


def main() -> None:
    cfg = tyro.cli(Union[
        Annotated[GptCfg,    tyro.conf.subcommand("gpt")],
        Annotated[PolicyCfg, tyro.conf.subcommand("policy")],
    ])

    tokens = _load_tensor_from_safetensors(cfg.tokens_path).to(torch.long)
    log.info(f"Loaded tokens {tuple(tokens.shape)}")

    if isinstance(cfg, GptCfg):
        # actions are just the next tokens — no model needed
        actions     = tokens[:, 1:].clone()
        num_latents = int(tokens.max().item()) + 1
        run_name    = "entropy_lb_gpt"
    else:
        if not cfg.actions_path:
            raise ValueError("--actions-path is required in policy mode")
        actions     = _load_tensor_from_safetensors(cfg.actions_path).to(torch.long)
        num_latents = cfg.num_latents
        run_name    = f"entropy_lb_policy_nlatents{num_latents}"
        log.info(f"Loaded actions {tuple(actions.shape)}")

    _run(
        tokens=tokens, actions=actions, num_latents=num_latents,
        pad_token_id=cfg.pad_token_id, max_steps=cfg.max_steps,
        run_name=run_name, wandb_cfg=cfg.wandb,
    )


if __name__ == "__main__":
    main()

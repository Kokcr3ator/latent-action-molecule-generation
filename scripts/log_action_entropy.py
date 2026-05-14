"""Compute the empirical conditional entropy H(A_t | S_{0:t}) and push it to
W&B as a horizontal reference line on the val/loss plot.

H(A | context) is a property of a (tokens, actions) pair — it does not depend
on any model, only on the action labels assigned to each molecule.  The actions
are produced by a specific LAM (via vq_encode), so you need to provide both
the tokenised dataset and the corresponding action dataset.

Usage:
    python -m scripts.log_action_entropy \\
        --tokens-path  /path/to/dataset.safetensors \\
        --actions-path /path/to/actions_dataset_num_latent_actions_2048_vocab_size_2048.safetensors \\
        --num-latents  2048 \\
        --wandb.group  exp6_distillation_impact

The script logs val/loss = H at every 100 steps up to --max-steps so the value
appears as a flat line when overlaid with policy distillation training curves.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

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
class Cfg:
    tokens_path: str                    # safetensors file with tokenised ZINC, shape [N, T]
    actions_path: str                   # safetensors file with action indices, shape [N, T-1]
    num_latents: int                    # codebook size used when generating the actions
    pad_token_id: int = 0
    max_steps: int = 20_000             # x-axis range for the horizontal line
    wandb: WandbCfg = field(default_factory=WandbCfg)


def main() -> None:
    cfg = tyro.cli(Cfg)

    tokens  = _load_tensor_from_safetensors(cfg.tokens_path).to(torch.long)
    actions = _load_tensor_from_safetensors(cfg.actions_path).to(torch.long)
    log.info(f"Loaded tokens {tuple(tokens.shape)}, actions {tuple(actions.shape)}")

    log.info("Computing H(A | S_{0:t})...")
    H = empirical_conditional_entropy(
        tokens=tokens,
        actions=actions,
        num_latents=cfg.num_latents,
        pad_token_id=cfg.pad_token_id,
    )
    H_max = math.log(cfg.num_latents)
    log.info(f"H(A | context) = {H:.4f} nats  (max = ln({cfg.num_latents}) = {H_max:.4f})")

    import wandb
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        name=f"entropy_lb_nlatents{cfg.num_latents}",
        dir=cfg.wandb.dir or None,
        config={"num_latents": cfg.num_latents, "conditional_entropy": H},
    )

    log.info(f"Pushing horizontal line at {H:.4f} nats for steps 0..{cfg.max_steps}...")
    for step in range(0, cfg.max_steps + 1, 100):
        wandb.log({"val/loss": H}, step=step)

    run.summary["dataset/conditional_entropy"] = H
    run.summary["dataset/marginal_entropy"]    = H_max
    wandb.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()

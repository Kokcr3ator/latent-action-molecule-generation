"""Compute the empirical conditional entropy H(A_t | S_{0:t}) for a ZINC dataset
and push it to W&B as a horizontal reference line on the val/loss plot.

Usage:
    python -m scripts.log_action_entropy \\
        --controllable-gpt-path /path/to/ckpt \\
        --wandb.group exp6_distillation_impact

The script logs val/loss = H at every step from 0 to --max-steps so the value
appears as a flat line when overlaid with policy distillation training curves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import tyro

from scripts.tokenise_dataset import run_tokenisation
from scripts.generate_actions import run_action_generation
from interdiff.dataset_entropy import empirical_conditional_entropy
from interdiff.models import ControllableGPT


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
    controllable_gpt_path: str          # path to a pretrained ControllableGPT checkpoint
    data_smiles: str = "interdiff/data/zinc/zinc.txt"
    vocab_size: int = 2048
    context_length: int = 128
    batch_size: int = 2048
    pad_token_id: int = 0
    max_steps: int = 20_000             # x-axis range for the horizontal line
    wandb: WandbCfg = field(default_factory=WandbCfg)


def main() -> None:
    cfg = tyro.cli(Cfg)

    # ------------------------------------------------------------------ data
    _, dataset_path = run_tokenisation(
        data_smiles=cfg.data_smiles,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
    )

    # ------------------------------------------------------------------ actions
    log.info("Generating actions from LAM...")
    cgpt = ControllableGPT.load(cfg.controllable_gpt_path)
    num_latents = cgpt.num_latents
    del cgpt  # free memory before generating actions

    from interdiff.io import _load_tensor_from_safetensors
    tokens = _load_tensor_from_safetensors(dataset_path).to(torch.long)

    actions = run_action_generation(
        controllable_gpt_path=cfg.controllable_gpt_path,
        dataset_path=dataset_path,
        batch_size=cfg.batch_size,
        pad_token_id=cfg.pad_token_id,
    )

    # ------------------------------------------------------------------ entropy
    log.info("Computing H(A | S_{0:t})...")
    H = empirical_conditional_entropy(
        tokens=tokens,
        actions=actions,
        num_latents=num_latents,
        pad_token_id=cfg.pad_token_id,
    )
    log.info(f"H(A | context) = {H:.4f} nats  (ln({num_latents}) = {torch.tensor(num_latents).float().log():.4f})")

    # ------------------------------------------------------------------ W&B
    import wandb
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        name=f"entropy_lb_nlatents{num_latents}",
        dir=cfg.wandb.dir or None,
        config={"num_latents": num_latents, "conditional_entropy": H},
    )

    # Log as a horizontal line by emitting the same value at every step
    log.info(f"Pushing {cfg.max_steps} steps to W&B as a horizontal line at {H:.4f}...")
    for step in range(0, cfg.max_steps + 1, 100):
        wandb.log({"val/loss": H, "step": step}, step=step)

    run.summary["dataset/conditional_entropy"] = H
    run.summary["dataset/marginal_entropy"] = float(torch.tensor(num_latents).float().log())
    wandb.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()

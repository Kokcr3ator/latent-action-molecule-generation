"""Supervised pretraining / policy-distillation training script.

Usage:
    python scripts/train.py --config configs/pretrain_base.yaml
    python scripts/train.py --config configs/pretrain_controllable.yaml
    python scripts/train.py --config configs/policy_distillation.yaml
    python scripts/train.py --config configs/pretrain_base.yaml --override training.batch_size=256 seed=123
"""
import argparse
import logging

import torch

from interdiff.config import load_config, instantiate, merge_with_overrides
from interdiff.utils.torch_utils import seed_all
from scripts.tokenise_dataset import run_tokenisation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from a YAML config.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config file.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Dotlist overrides, e.g. training.batch_size=256 seed=123")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load and optionally override config
    cfg = load_config(args.config)
    cfg = merge_with_overrides(cfg, args.override)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("train")

    seed_all(cfg.seed)

    save_path = run_tokenisation(cfg)  # returns path to dataset.safetensors
    log.info(f"Using tokenised dataset at {save_path}")

    # Instantiate components from config
    model = instantiate(cfg.model)
    train_cfg = instantiate(cfg.training)
    optim = instantiate(cfg.optimizer, model=model)
    sched = instantiate(cfg.scheduler, optimizer=optim)
    logger = instantiate(cfg.log) if bool(cfg.wandb_log) else None

    train_cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = instantiate(cfg.trainer, model=model, scheduler=sched, optimizer=optim, logger=logger, train_cfg=train_cfg)

    loaders = instantiate(cfg.loader, dataset_path=save_path)
    train_dataloader = loaders.train_loader
    val_dataloader = loaders.val_loader

    trainer.fit(train_dataloader, val_dataloader)

    if logger:
        logger.finalize()


if __name__ == "__main__":
    main()

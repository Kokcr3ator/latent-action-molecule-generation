"""Utilities for reporting model parameter counts."""
from __future__ import annotations

from typing import Any, Dict

from omegaconf import OmegaConf, DictConfig
from torch import nn


def parameter_counts(model: nn.Module) -> Dict[str, int | float]:
    """Return parameter-count statistics for a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    counts: Dict[str, int | float] = {
        "num_parameters": total,
        "num_parameters_millions": total / 1e6,
        "num_trainable_parameters": trainable,
        "num_trainable_parameters_millions": trainable / 1e6,
        "num_frozen_parameters": total - trainable,
        "num_frozen_parameters_millions": (total - trainable) / 1e6,
    }

    get_num_params = getattr(model, "get_num_params", None)
    if callable(get_num_params):
        non_embedding = int(get_num_params())
        counts["num_non_embedding_parameters"] = non_embedding
        counts["num_non_embedding_parameters_millions"] = non_embedding / 1e6

    return counts


def namespaced_parameter_counts(model: nn.Module, namespace: str) -> Dict[str, int | float]:
    """Return parameter counts with wandb-friendly namespaced keys."""
    return {
        f"{namespace}/{key}": value
        for key, value in parameter_counts(model).items()
    }


def log_parameter_counts(logger: Any, model: nn.Module, namespace: str) -> Dict[str, int | float]:
    """Log parameter counts to the run config (static metadata, not a chart metric)."""
    counts = namespaced_parameter_counts(model, namespace)
    logger.log_config(counts)
    return counts


def log_run_setup(
    logger: Any,
    cfg: DictConfig,
    config_file: str | None = None,
    overrides: list[str] | None = None,
    **models: nn.Module,
) -> None:
    """Log everything needed to reproduce a run: config, parameter counts, and num_latents.

    Call once after the logger and all models are instantiated. Logs nothing when
    logger is None so callers do not need to guard the call themselves.

    Args:
        logger: W&B (or other) logger instance, or None.
        cfg: Resolved OmegaConf config for the run.
        config_file: Path to the config file passed on the command line.
        overrides: List of dotlist override strings passed on the command line.
        **models: Keyword-named models to log, e.g. model=model, ppo_agent=ppo_agent.
                  Each is logged under its keyword name as the namespace.
    """
    if logger is None:
        return
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    logger.log_config(cfg_dict)
    run_meta: Dict[str, Any] = {}
    if config_file is not None:
        run_meta["run/config_file"] = config_file
    if overrides:
        run_meta["run/overrides"] = " ".join(overrides)
    if run_meta:
        logger.log_config(run_meta)
    for namespace, model in models.items():
        log_parameter_counts(logger, model, namespace)
        if hasattr(model, "num_latents"):
            logger.log_config({f"{namespace}/num_latents": model.num_latents})

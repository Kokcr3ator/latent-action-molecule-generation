"""Utilities for reporting model parameter counts."""
from __future__ import annotations

from typing import Any, Dict, Union

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


def log_parameter_counts(logger: Any, model: nn.Module, namespace: str) -> Dict[str, int | float]:
    """Log total parameter count (millions) to the run config as a single number."""
    counts = parameter_counts(model)
    total_M = counts["num_parameters_millions"]
    logger.log_config({f"{namespace}/parameters_M": round(total_M, 3)})
    return counts


def log_run_setup(
    logger: Any,
    cfg: Union[Dict, Any],
    **models: nn.Module,
) -> None:
    """Log config and parameter counts. No-op when logger is None."""
    if logger is None:
        return
    cfg_dict = cfg if isinstance(cfg, dict) else vars(cfg)
    logger.log_config(cfg_dict)
    for namespace, model in models.items():
        log_parameter_counts(logger, model, namespace)

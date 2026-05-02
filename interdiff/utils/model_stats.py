"""Utilities for reporting model parameter counts."""
from __future__ import annotations

from typing import Any, Dict

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
    """Log parameter counts to a logger config and as a one-time metric."""
    counts = namespaced_parameter_counts(model, namespace)
    logger.log_config(counts)
    logger.log({**counts, "step": 0})
    return counts

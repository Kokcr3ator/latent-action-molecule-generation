"""Configuration utilities replacing Hydra with simple YAML + OmegaConf.

Provides:
    - load_config(path) -- load a YAML file and resolve OmegaConf interpolations
    - instantiate(cfg)  -- create objects from a config node that contains ``_target_``
    - merge_with_overrides(cfg, overrides) -- apply CLI-style overrides to a config
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

from omegaconf import OmegaConf, DictConfig

# ---------------------------------------------------------------------------
# Registry: maps ``_target_`` strings to the actual callables
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Any] = {}


def _resolve_target(target: str) -> Any:
    """Import and return the object referred to by *target* (dot-separated path).

    Results are cached in ``_REGISTRY`` for fast repeat look-ups.
    """
    if target in _REGISTRY:
        return _REGISTRY[target]
    module_path, _, attr_name = target.rpartition(".")
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name)
    _REGISTRY[target] = obj
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str) -> DictConfig:
    """Load a YAML file and return a fully-resolved OmegaConf ``DictConfig``.

    OmegaConf interpolations (``${...}``) are resolved lazily on access.
    """
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return cfg


def instantiate(cfg: Any, **kwargs: Any) -> Any:
    """Create an object from a config node containing a ``_target_`` key.

    Extra ``**kwargs`` are forwarded to the constructor and override any
    identically-named keys that appear in the config.

    If *cfg* is a plain ``DictConfig`` / dict *without* ``_target_``, a
    ``ValueError`` is raised.

    Nested ``_target_`` nodes are **not** automatically instantiated — only
    the top level is resolved. This keeps behaviour predictable and avoids
    hidden side-effects.
    """
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        raise TypeError(f"Expected DictConfig or dict, got {type(cfg)}")

    target = cfg_dict.pop("_target_", None)
    if target is None:
        raise ValueError(
            "Cannot instantiate a config without a '_target_' key. "
            f"Keys present: {list(cfg_dict.keys())}"
        )

    # Merge caller-supplied kwargs (they win over config values)
    cfg_dict.update(kwargs)

    cls_or_fn = _resolve_target(target)
    return cls_or_fn(**cfg_dict)


def merge_with_overrides(cfg: DictConfig, overrides: List[str]) -> DictConfig:
    """Apply dotlist-style overrides to *cfg* and return the merged result.

    Each element of *overrides* should be of the form ``key=value``, where
    nested keys use dots (e.g. ``training.batch_size=256``).

    The returned ``DictConfig`` is a new object; *cfg* is not mutated.
    """
    if not overrides:
        return cfg
    override_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, override_cfg)


def to_absolute_path(path: str) -> str:
    """Return *path* unchanged (compatibility shim for code that used Hydra's
    ``to_absolute_path``).  With self-contained configs all paths are already
    relative to the project root or absolute."""
    import os
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)

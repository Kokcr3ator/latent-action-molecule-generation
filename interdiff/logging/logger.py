from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from torch import nn


class Logger(ABC):
    @abstractmethod
    def log(self, metrics: dict): ...
    def log_config(self, config: dict): pass
    def log_artifact(self, path: str, name: str, type: str = "model"): pass
    @abstractmethod
    def log_table(self, table_name: str, columns: List[str], data: List[List], step: Optional[int] = None): ...
    @abstractmethod
    def finalize(self): ...


def log_parameter_counts(logger: Logger, model: nn.Module, namespace: str) -> None:
    """Log total parameter count (millions) to the run config."""
    from interdiff.metrics import parameter_counts
    counts = parameter_counts(model)
    logger.log_config({f"{namespace}/parameters_M": round(counts["num_parameters_millions"], 3)})


def log_run_setup(logger: Optional[Logger], cfg: Union[Dict, Any], **models: nn.Module) -> None:
    """Log config dict and parameter counts for each model. No-op when logger is None."""
    if logger is None:
        return
    cfg_dict = cfg if isinstance(cfg, dict) else vars(cfg)
    logger.log_config(cfg_dict)
    for namespace, model in models.items():
        log_parameter_counts(logger, model, namespace)

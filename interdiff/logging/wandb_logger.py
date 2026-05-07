from typing import List, Optional

import wandb
from interdiff.logging.logger import Logger


class WandbLogger(Logger):
    def __init__(self, project: str, name: str, group: str, entity: Optional[str] = None):
        wandb.init(project=project, name=name, group=group, entity=entity)

    def log(self, metrics: dict):
        step = metrics.get("step")
        wandb.log(metrics, step=step)

    def log_config(self, config: dict):
        wandb.config.update(config)

    def log_table(self, table_name: str, columns: List[str], data: List[List], step: Optional[int] = None):
        wandb.log({table_name: wandb.Table(columns=columns, data=data)}, step=step)

    def finalize(self):
        wandb.finish()

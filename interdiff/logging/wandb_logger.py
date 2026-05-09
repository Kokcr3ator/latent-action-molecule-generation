import os
from typing import List, Optional

import wandb
from interdiff.logging.logger import Logger


class WandbLogger(Logger):
    def __init__(self, project: str, name: str, group: str, entity: Optional[str] = None, dir: Optional[str] = None):
        if dir:
            os.makedirs(dir, exist_ok=True)
        wandb.init(project=project, name=name, group=group, entity=entity, dir=dir or None)

    def log(self, metrics: dict):
        wandb.log(metrics)

    def log_config(self, config: dict):
        wandb.config.update(config)

    def log_artifact(self, path: str, name: str, type: str = "model"):
        artifact = wandb.Artifact(name=name, type=type)
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def log_table(self, table_name: str, columns: List[str], data: List[List], step: Optional[int] = None):
        wandb.log({table_name: wandb.Table(columns=columns, data=data)}, step=step)

    def finalize(self):
        wandb.finish()

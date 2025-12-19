import wandb
from interdiff.logging.logger import Logger

class WandbLogger(Logger):
    def __init__(self, project: str, name: str):
        wandb.init(project=project, name=name)
    def log(self, metrics: dict):
        wandb.log(metrics)
    def finalize(self):
        wandb.finish()
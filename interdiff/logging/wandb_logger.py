from typing import List, Optional

import wandb
from interdiff.logging.logger import Logger

class WandbLogger(Logger):
    def __init__(self, project: str, name: str, group: str, entity: Optional[str] = None):
        wandb.init(project=project, name=name, group=group, entity=entity)
    
    def log(self, metrics: dict):
        wandb.log(metrics)
    
    def log_config(self, config: dict):
        wandb.config.update(config)

    def log_table(self, table_name: str, columns: List[str], data: List[List], step: Optional[int] = None):
        """Log a table to wandb.
        
        Args:
            table_name: Name of the table to log.
            columns: List of column names.
            data: List of rows, where each row is a list of values.
            step: Optional step number to associate with the table.
        """
        table = wandb.Table(columns=columns, data=data)
        log_dict = {table_name: table}
        if step is not None:
            log_dict["step"] = step
        wandb.log(log_dict)
    
    def finalize(self):
        wandb.finish()
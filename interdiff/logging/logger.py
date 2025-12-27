from abc import ABC, abstractmethod
from typing import List, Optional

class Logger(ABC):
    @abstractmethod
    def log(self, metrics: dict): ...
    @abstractmethod
    def log_table(self, table_name: str, columns: List[str], data: List[List], step: Optional[int] = None): ...
    @abstractmethod
    def finalize(self): ...

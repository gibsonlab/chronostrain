from .base import AbstractConfig
from .database import DatabaseConfig
from .model import ModelConfig
from .torch import TorchConfig
from .external import ExternalToolsConfig
from .entrez import EntrezConfig


class ChronostrainConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ChronoStrain", cfg)
        self.database_cfg: DatabaseConfig = DatabaseConfig(
            cfg=self.get_item("Database"),
            database_kwargs=self.get_item("Database.args")
        )
        self.model_cfg: ModelConfig = ModelConfig(self.get_item("Model"))
        self.torch_cfg: TorchConfig = TorchConfig(self.get_item("PyTorch"))
        self.external_tools_cfg: ExternalToolsConfig = ExternalToolsConfig(self.get_item("ExternalTools"))
        self.entrez_cfg: EntrezConfig = EntrezConfig(self.get_item("Entrez"))

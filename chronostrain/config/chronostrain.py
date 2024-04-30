from .base import AbstractConfig
from .database import DatabaseConfig
from .model import ModelConfig
from .engine import EngineConfig
from .external import ExternalToolsConfig
from .entrez import EntrezConfig


class ChronostrainConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ChronoStrain", cfg)
        self.database_cfg: DatabaseConfig = DatabaseConfig(
            cfg=self.get_item("Database"),
            database_kwargs=self.get_item("Database.ParserArgs")
        )
        self.model_cfg: ModelConfig = ModelConfig(self.get_item("Model"))
        self.engine_cfg: EngineConfig = EngineConfig(self.get_item("Engine"))
        self.external_tools_cfg: ExternalToolsConfig = ExternalToolsConfig(self.get_item("ExternalTools"))
        self.entrez_cfg: EntrezConfig = EntrezConfig(self.get_item("Entrez"))

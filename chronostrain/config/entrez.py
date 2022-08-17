from .base import AbstractConfig
from ..util.entrez import init_entrez


class EntrezConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Entrez", cfg)
        self.email = self.get_str("EMAIL")
        init_entrez(self.email)

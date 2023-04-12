from .base import AbstractConfig


class EntrezConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Entrez", cfg)
        self.email = self.get_str("EMAIL")

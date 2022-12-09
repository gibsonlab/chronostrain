from .base import AbstractConfig


class ExternalToolsConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ExternalTools", cfg)
        self.pairwise_align_cmd = self.get_str("PAIRWISE_ALN_BACKEND")

from .base import AbstractConfig


class ExternalToolsConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("ExternalTools", cfg)
        self.pairwise_align_cmd = self.get_str("PAIRWISE_ALN_BACKEND")
        self.pairwise_align_use_bam = self.get_bool("PAIRWISE_ALN_BAM", default_value=True)

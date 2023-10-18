from pathlib import Path
from .base import AbstractConfig


class ModelConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Model", cfg)
        self.use_quality_scores: bool = self.get_bool("USE_QUALITY_SCORES")
        self.num_cores: int = self.get_int("NUM_CORES")
        self.cache_dir: Path = self.get_path("CACHE_DIR")
        self.cache_enabled: bool = self.get_bool("CACHE_ENABLED", default_value=True)
        self.sics_dof_1: float = self.get_float("SICS_DOF_1")
        self.sics_scale_1: float = self.get_float("SICS_SCALE_1")
        self.sics_dof: float = self.get_float("SICS_DOF")
        self.sics_scale: float = self.get_float("SICS_SCALE")
        self.inverse_temperature: float = self.get_float("INV_TEMPERATURE")
        self.use_sparse: bool = self.get_bool("SPARSE_MATRICES")
        self.min_overlap_ratio: float = self.get_float("MIN_OVERLAP_RATIO")

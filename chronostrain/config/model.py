from pathlib import Path
from .base import AbstractConfig


class ModelConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Model", cfg)
        self.use_quality_scores: bool = self.get_bool("USE_QUALITY_SCORES")
        self.num_cores: int = self.get_int("NUM_CORES")
        self.cache_dir: Path = self.get_path("CACHE_DIR")
        self.sics_dof_1: float = self.get_float("SICS_DOF_1")
        self.sics_scale_1: float = self.get_float("SICS_SCALE_1")
        self.sics_dof: float = self.get_float("SICS_DOF")
        self.sics_scale: float = self.get_float("SICS_SCALE")
        self.use_sparse: bool = self.get_bool("SPARSE_MATRICES")
        self.frag_len_negbin_n: float = self.get_float("FRAG_LEN_NB_N")
        self.frag_len_negbin_p: float = self.get_float("FRAG_LEN_NB_P")
        self.min_overlap_ratio: float = self.get_float("MIN_OVERLAP_RATIO")
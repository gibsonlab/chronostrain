"""
 initialize.py
 Contains implementations of the proposed algorithms.
"""
from abc import ABCMeta, abstractmethod

from chronostrain.database import StrainDatabase
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from chronostrain.algs.subroutines.likelihoods import SparseDataLikelihoods

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase):
        self.model = model
        self.data = data
        self.db = db

    @property
    def data_likelihoods(self) -> SparseDataLikelihoods:
        return SparseDataLikelihoods(
            self.model, self.data, self.db, num_cores=cfg.model_cfg.num_cores, dtype=cfg.engine_cfg.dtype
        )

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass




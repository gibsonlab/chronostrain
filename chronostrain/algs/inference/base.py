"""
 base.py
 Contains implementations of the proposed algorithms.
"""
from abc import ABCMeta, abstractmethod
from chronostrain.database import StrainDatabase
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from chronostrain.algs.subroutines import DenseDataLikelihoods, SparseDataLikelihoods

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self, model: GenerativeModel, data: TimeSeriesReads, db: StrainDatabase):
        self.model = model
        self.data = data
        if cfg.model_cfg.use_sparse:
            self.data_likelihoods = SparseDataLikelihoods(model, data, db)
        else:
            self.data_likelihoods = DenseDataLikelihoods(model, data)

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass




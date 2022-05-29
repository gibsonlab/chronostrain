"""
 base.py
 Contains implementations of the proposed algorithms.
"""
from abc import ABCMeta, abstractmethod
from typing import Optional

from chronostrain.database import StrainDatabase
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from chronostrain.algs.subroutines.likelihoods import DenseDataLikelihoods, SparseDataLikelihoods, DataLikelihoods

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 num_cores: int = 1,
                 precomputed_data_likelihoods: Optional[DataLikelihoods] = None):
        self.model = model
        self.data = data
        if precomputed_data_likelihoods is not None:
            self.data_likelihoods = precomputed_data_likelihoods
        else:
            if cfg.model_cfg.use_sparse:
                self.data_likelihoods = SparseDataLikelihoods(
                    model, data, db, num_cores=num_cores
                )
            else:
                self.data_likelihoods = DenseDataLikelihoods(model, data)

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass




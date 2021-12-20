"""
 base.py
 Contains implementations of the proposed algorithms.
"""
from abc import ABCMeta, abstractmethod
from chronostrain.database import StrainDatabase
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from chronostrain.algs.subroutines.likelihoods import DenseDataLikelihoods, SparseDataLikelihoods

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 frag_chunk_size: int,
                 num_cores: int = 1):
        self.model = model
        self.data = data
        self.frag_chunk_size = frag_chunk_size
        if cfg.model_cfg.use_sparse:
            self.data_likelihoods = SparseDataLikelihoods(
                model, data, db, num_cores=num_cores, frag_chunk_size=frag_chunk_size
            )
        else:
            self.data_likelihoods = DenseDataLikelihoods(model, data)

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass




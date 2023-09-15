"""
 initialize.py
 Contains implementations of the proposed algorithms.
"""
from abc import ABCMeta, abstractmethod

from chronostrain.database import StrainDatabase
from chronostrain.model import TimeSeriesReads, AbstractErrorModel, AbundanceGaussianPrior
from chronostrain.logging import create_logger

logger = create_logger(__name__)


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self,
                 gaussian_prior: AbundanceGaussianPrior,
                 error_model: AbstractErrorModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase):
        self.gaussian_prior: AbundanceGaussianPrior = gaussian_prior
        self.error_model = error_model
        self.data: TimeSeriesReads = data
        self.db: StrainDatabase = db

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

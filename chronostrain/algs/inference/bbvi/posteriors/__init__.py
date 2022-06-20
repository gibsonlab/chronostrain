from .base import AbstractReparametrizedPosterior
from .gaussians import GaussianPosteriorStrainCorrelation, \
    GaussianPosteriorFullReparametrizedCorrelation, \
    GaussianPosteriorTimeCorrelation, \
    GaussianPosteriorAutoregressiveReparametrizedCorrelation
from .dirichlet import ReparametrizedDirichletPosterior

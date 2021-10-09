from chronostrain.config import cfg
from chronostrain.model import Population
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.data_cache import ComputationCache, CacheTag


class LikelihoodMatrixCache(ComputationCache):
    """
    A subclass of ComputationCache representing the results of likelihood matrix computation
     associated with a particular TimeSeriesReads instance and collection of Strains (variants).

    The specific cache dependency is:
    1) Configuration's use_quality_scores flag,
    2) The bacterial population, specified by a list of repr(strain) for each strain in the population.
    3) The read file paths, in order of specification (if the order changes, then so does the cache key.)
    """
    def __init__(self, reads: TimeSeriesReads, pop: Population):
        super().__init__(
            CacheTag(
                use_quality=cfg.model_cfg.use_quality_scores,
                strains=[
                    repr(strain) for strain in pop.strains
                ],
                file_paths=[reads_t.src.paths for reads_t in reads]  # read files
            )
        )

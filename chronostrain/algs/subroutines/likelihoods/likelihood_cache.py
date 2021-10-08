from chronostrain.config import cfg
from chronostrain.model import Population
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.data_cache import ComputationCache, CacheTag


class LikelihoodMatrixCache(ComputationCache):
    """
    A subclass of ComputationCache representing the results of likelihood matrix computation
     associated with a particular TimeSeriesReads instance and collection of Strains (variants).
    """
    def __init__(self, reads: TimeSeriesReads, pop: Population):
        super().__init__(
            CacheTag(
                use_quality=cfg.model_cfg.use_quality_scores,
                markers=[
                    repr(strain) for strain in pop.strains
                ],
                file_paths=[reads_t.src.paths for reads_t in reads]  # read files
            )
        )

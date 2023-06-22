from chronostrain.database import StrainDatabase
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.config import cfg
from chronostrain.model.io import TimeSeriesReads


class ReadsPopulationCache(ComputationCache):
    """
    A subclass of ComputationCache representing the results of likelihood matrix computation
     associated with a particular TimeSeriesReads instance and collection of Strains (variants).

    The specific cache dependency is:
    1) Configuration's use_quality_scores flag,
    2) The bacterial population, specified by a list of repr(strain) for each strain in the population.
    3) The read file paths, in order of specification (if the order changes, then so does the cache key.)
    """
    def __init__(self, reads: TimeSeriesReads, db: StrainDatabase):
        super().__init__(
            CacheTag(
                use_quality=cfg.model_cfg.use_quality_scores,
                database=db.signature,
                file_paths=[
                    src.path
                    for reads_t in reads
                    for src in reads_t.sources
                ]  # read files
            )
        )

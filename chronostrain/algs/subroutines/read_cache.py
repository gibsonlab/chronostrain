from chronostrain.config import cfg
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.data_cache import CacheTag, ComputationCache


class ReadsComputationCache(ComputationCache):
    """
    A subclass of ComputationCache representing the results of all computation associated with a particular
    TimeSeriesReads instance (or equivalent instantiations).
    """
    def __init__(self, reads: TimeSeriesReads):
        super().__init__(
            CacheTag(
                file_paths=[reads_t.src.paths for reads_t in reads]  # read files
            )
        )

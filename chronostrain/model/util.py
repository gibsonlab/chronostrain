from typing import Iterator, Tuple
from chronostrain.util.sequences import Sequence, AllocatedSequence
from chronostrain.model import FragmentSpace
from .bacteria import Population

import numpy as np
from chronostrain.logging import create_logger
logger = create_logger(__name__)


def sliding_window_bytes(seq: Sequence, width: int) -> Iterator[Tuple[np.ndarray, int]]:
    """
    A generator for the subsequences produced by a sliding window of specified width.
    """
    for i in range(len(seq) - width + 1):
        yield seq.bytes[i:i + width], i


def construct_fragment_space_uniform_length(window_size: int, population: Population) -> FragmentSpace:
    """
        Retrieves the fragment space via lazy instantiation.
        Returns a FragmentSpace instance.
    """
    logger.debug("Constructing fragment space for window size {}...".format(window_size))
    fragment_space = FragmentSpace()
    for strain in population.strains:
        for marker in strain.markers:
            for seq_bytes, pos in sliding_window_bytes(marker.seq, window_size):
                fragment_space.add_seq(
                    AllocatedSequence(seq_bytes),
                    metadata="{}_{}_Pos({})".format(strain.id, marker.id, pos)
                )

    logger.debug("Finished constructing fragment space. (Size={})".format(len(fragment_space)))
    return fragment_space

from typing import Iterator, Tuple
from chronostrain.util.sequences import SeqType
from .fragments import FragmentSpace
from .bacteria import Population

from chronostrain.logging import create_logger
logger = create_logger(__name__)


def sliding_window(seq: SeqType, width: int) -> Iterator[Tuple[SeqType, int]]:
    """
    A generator for the subsequences produced by a sliding window of specified width.
    """
    for i in range(len(seq) - width + 1):
        yield seq[i:i + width], i


def construct_fragment_space_uniform_length(window_size: int, population: Population) -> FragmentSpace:
    """
        Retrieves the fragment space via lazy instantiation.
        Returns a FragmentSpace instance.
    """
    logger.debug("Constructing fragment space for window size {}...".format(window_size))
    fragment_space = FragmentSpace()
    for strain in population.strains:
        for marker in strain.markers:
            for seq, pos in sliding_window(marker.seq, window_size):
                fragment_space.add_seq(seq, metadata="{}_{}_Pos({})".format(strain.id, marker.id, pos))

    logger.debug("Finished constructing fragment space. (Size={})".format(fragment_space.size()))
    return fragment_space

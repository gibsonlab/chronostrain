from logging import Logger
from chronostrain.algs.subroutines.alignments import CachedReadPairwiseAlignments, CachedReadMultipleAlignments
from chronostrain.algs.subroutines.cache import ReadsComputationCache
from chronostrain.database import StrainDatabase
from chronostrain.model import FragmentSpace
from chronostrain.config import cfg
from chronostrain.model.io import TimeSeriesReads


def aligned_exact_fragments(reads: TimeSeriesReads, db: StrainDatabase, logger: Logger, mode: str = 'pairwise') -> FragmentSpace:
    logger.info("Constructing fragments from alignments.")
    fragment_space = FragmentSpace()

    if mode == 'pairwise':
        alignments = CachedReadPairwiseAlignments(reads, db)
        for _, aln in alignments.get_alignments():
            # First, add the likelihood for the fragment for the aligned base marker.
            fragment_space.add_seq(
                aln.marker_frag
            )

            if len(aln.marker_frag) < 15:
                raise Exception("UNEXPECTED ERROR! found frag of length smaller than 15")
    elif mode == 'multiple':
        multiple_alignments = CachedReadMultipleAlignments(reads, db)
        for multi_align in multiple_alignments.get_alignments(num_cores=cfg.model_cfg.num_cores):
            logger.debug(f"Constructing fragments for marker `{multi_align.canonical_marker.name}`.")

            for frag_entry in multi_align.all_mapped_fragments():
                marker, read, subseq, insertions, deletions, start_clip, end_clip, revcomp = frag_entry

                fragment_space.add_seq(
                    subseq,
                    metadata=f"({read.id}->{marker.id})"
                )
    else:
        raise ValueError(f"Unknown fragment extrapolation mode `{mode}`.")
    return fragment_space


def load_fragments(reads: TimeSeriesReads, db: StrainDatabase, logger: Logger) -> FragmentSpace:
    cache = ReadsComputationCache(reads)
    return cache.call(
        relative_filepath="inference_fragments.pkl",
        fn=aligned_exact_fragments,
        call_args=[reads, db, logger]
    )

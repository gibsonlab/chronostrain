from logging import Logger
from pathlib import Path

from chronostrain.algs.subroutines.alignments import CachedReadPairwiseAlignments
from chronostrain.algs.subroutines.cache import ReadsPopulationCache
from chronostrain.database import StrainDatabase
from chronostrain.model import FragmentSpace, UnallocatedFragmentSpace
from chronostrain.model.io import TimeSeriesReads


def aligned_exact_fragments(reads: TimeSeriesReads,
                            db: StrainDatabase,
                            n_threads: int,
                            logger: Logger,
                            mode: str = 'pairwise') -> FragmentSpace:
    logger.info("Constructing fragments from alignments.")
    fragment_space = FragmentSpace()

    if mode == 'pairwise':
        alignments = CachedReadPairwiseAlignments(reads, db, n_threads=n_threads)
        for t_idx, aln in alignments.get_alignments():
            # First, add the likelihood for the fragment for the aligned base marker.
            frag = fragment_space.add_seq(aln.marker_frag)
            if frag.metadata is None or len(frag.metadata) == 0:
                frag.add_metadata(f"{aln.sam_path.name}@L{aln.sam_line_no}")
            if len(aln.marker_frag) < 15:
                raise Exception("UNEXPECTED ERROR! found frag of length smaller than 15")
    elif mode == 'multiple':
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown fragment extrapolation mode `{mode}`.")
    return fragment_space


def aligned_exact_fragments_dynamic(reads: TimeSeriesReads,
                                    db: StrainDatabase,
                                    work_dir: Path,
                                    n_threads: int,
                                    logger: Logger, mode: str = 'pairwise') -> UnallocatedFragmentSpace:
    logger.info("Constructing fragments from alignments (disk-allocation).")
    fasta_path = work_dir / "fragments.fasta"
    fragment_space = UnallocatedFragmentSpace(fasta_path=fasta_path)

    if mode == 'pairwise':
        alignments = CachedReadPairwiseAlignments(reads, db, n_threads=n_threads)
        for t_idx, aln in alignments.get_alignments():
            frag = fragment_space.add_seq(aln.marker_frag)
            if len(frag.metadata) == 0:
                frag.add_metadata(f"{aln.sam_path.name}@L{aln.sam_line_no}")
            if len(aln.marker_frag) < 15:
                raise Exception("UNEXPECTED ERROR! found frag of length smaller than 15")

    fragment_space.write_fasta_records()
    return fragment_space


def load_fragments(reads: TimeSeriesReads, db: StrainDatabase, n_threads: int, logger: Logger) -> FragmentSpace:
    cache = ReadsPopulationCache(reads, db)
    return cache.call(
        relative_filepath="inference_fragments.pkl",
        fn=aligned_exact_fragments,
        call_args=[reads, db, n_threads, logger]
    )


def load_fragments_dynamic(reads: TimeSeriesReads, db: StrainDatabase, n_threads: int, logger: Logger) -> UnallocatedFragmentSpace:
    cache = ReadsPopulationCache(reads, db)
    frags: UnallocatedFragmentSpace = cache.call(
        relative_filepath="inference_fragments_dynamic.pkl",
        fn=aligned_exact_fragments_dynamic,
        call_args=[reads, db, cache.cache_dir, n_threads, logger]
    )
    if not frags.fasta_resource.fasta_path.exists():
        # Cache is corrupted; clear and recompute
        p = cache.cache_dir / "inference_fragments_dynamic.pkl"
        p.unlink()
        frags: UnallocatedFragmentSpace = cache.call(
            relative_filepath="inference_fragments_dynamic.pkl",
            fn=aligned_exact_fragments_dynamic,
            call_args=[reads, db, cache.cache_dir, n_threads, logger]
        )

    return frags

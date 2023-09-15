import itertools
from dataclasses import dataclass
from typing import Iterator, List, Tuple
from pathlib import Path

import jax.numpy as jnp
import numpy as cnp
import pandas as pd
import jax.experimental.sparse as jsparse

from chronostrain.model import Fragment, FragmentReadErrorLikelihood, FragmentSpace, FragmentPairSpace, \
    TimeSeriesReads, AbstractErrorModel, TimeSliceReads, ReadType, SampleReadSourcePaired, SampleReadSourceSingle, \
    SequenceRead, Marker, PairedEndRead
from chronostrain.database import StrainDatabase
from chronostrain.util.alignments.pairwise import SequenceReadPairwiseAlignment

from .cache import ReadStrainCollectionCache
from .read_order import UniqueReadOrdering, UniquePairedReadOrdering
from .cached_pairwise_alignment import CachedReadPairwiseAlignments

from chronostrain.config import cfg
from chronostrain.logging import create_logger

logger = create_logger(__name__)


@dataclass
class TimeSliceLikelihoods:
    lls: FragmentReadErrorLikelihood
    paired_lls: FragmentReadErrorLikelihood

    def diagnostic(self):
        _, counts_per_read = jnp.unique(self.lls.matrix.indices[:, 1], return_counts=True)
        logger.debug(
            "Read-likelihood matrix (size {r} x {c}) has {nz} nonzero entries. "
            "(~{meanct:.2f}±{stdct:.2f} hits per read)".format(
                r=self.lls.matrix.shape[0], c=self.lls.matrix.shape[1],
                nz=len(self.lls.matrix.data),
                meanct=counts_per_read.mean().item(),
                stdct=counts_per_read.std().item()
            )
        )
        _, counts_per_read_pair = jnp.unique(self.paired_lls.matrix.indices[:, 1], return_counts=True)
        logger.debug(
            "Paired Read-likelihood matrix (size {r} x {c}) has {nz} nonzero entries. "
            "(~{meanct:.2f}±{stdct:.2f} hits per paired read)".format(
                r=self.paired_lls.matrix.shape[0], c=self.paired_lls.matrix.shape[1],
                nz=len(self.paired_lls.matrix.data),
                meanct=counts_per_read_pair.mean().item(),
                stdct=counts_per_read_pair.std().item()
            )
        )

    def save(self, out_dir: Path):
        self.lls.save(out_dir / 'lls.npz')
        self.paired_lls.save(out_dir / 'lls_paired.npz')

    @staticmethod
    def load(base_dir: Path):
        base_dir.mkdir(exist_ok=True, parents=True)
        lls = FragmentReadErrorLikelihood.load(base_dir / 'lls.npz')
        paired_lls = FragmentReadErrorLikelihood.load(base_dir / 'lls_paired.npz')
        return TimeSliceLikelihoods(lls, paired_lls)


@dataclass
class TimeSeriesLikelihoods:
    slices: List[TimeSliceLikelihoods]
    fragments: FragmentSpace
    fragment_pairs: FragmentPairSpace

    def save(self, out_dir: Path):
        out_dir.mkdir(exist_ok=True, parents=True)
        for t_idx, t_lls in enumerate(self.slices):
            t_lls.save(out_dir / 'log_likelihoods' / str(t_idx))
        self.fragments.save(out_dir / 'fragments' / 'inference_fragments.pkl')
        self.fragment_pairs.save(out_dir / 'fragments' / 'inference_fragment_pairs.pkl')

    @staticmethod
    def load(num_slices: int, base_dir: Path) -> 'TimeSeriesLikelihoods':
        _slices = [
            TimeSliceLikelihoods.load(base_dir / 'log_likelihoods' / str(k))
            for k in range(num_slices)
        ]
        _frags = FragmentSpace.load(base_dir / 'fragments' / 'inference_fragments.pkl')
        _frag_pairs = FragmentPairSpace.load(base_dir / 'fragments' / 'inference_fragment_pairs.pkl')
        return TimeSeriesLikelihoods(_slices, _frags, _frag_pairs)


class ReadFragmentMappings:
    def __init__(
            self,
            reads: TimeSeriesReads,
            db: StrainDatabase,
            error_model: AbstractErrorModel,
            dtype: str,
            ll_threshold: float = -100.0
    ):
        self.reads: TimeSeriesReads = reads
        self.db = db
        self.error_model = error_model
        self.ll_threshold = ll_threshold
        self.add_frag_metadata = True
        self.dtype = dtype

        self.cache = ReadStrainCollectionCache(reads, db)
        self.alignment_wrapper = CachedReadPairwiseAlignments(
            self.reads, self.db,
            cache=self.cache,
            n_threads=cfg.model_cfg.num_cores,
            print_tqdm=True
        )
        self.model_values: TimeSeriesLikelihoods = self.compute_all_cached()

    def compute_all_cached(self) -> TimeSeriesLikelihoods:
        def _save(p: Path, res: TimeSeriesLikelihoods):
            if p.exists() and p.is_file():
                raise ValueError(
                    "TimeSeriesLikelihoods save() method must specify a directory."
                    f"Instead, found a file with the same name as destination: {p}"
                )
            res.save(p)

        return self.cache.call(
            relative_filepath='.',
            fn=self.compute_all,
            save=_save,
            load=lambda p: TimeSeriesLikelihoods.load(len(self.reads), p)
        )

    def compute_all(self) -> TimeSeriesLikelihoods:
        all_matrices = []
        fragments = FragmentSpace()
        fragment_pairs = FragmentPairSpace()

        results = []
        # Do the loop once to populate the entire fragment/fragment pair index.
        # Before this first loop is done, len(fragments) won't return the correct value.
        for t_idx, reads_t in enumerate(self.reads):
            logger.debug("Processing alignments for timepoint t={} ({} of {})".format(
                reads_t.time_point, t_idx + 1, len(self.reads))
            )
            results.append(self.process_time_slice(reads_t, fragments, fragment_pairs))

        for t_idx, (ll_values, paired_ll_values, indices, paired_indices, n_reads, n_read_pairs) in enumerate(results):
            logger.debug(f"Instantiating matrix for t_idx = {t_idx}")
            lls = FragmentReadErrorLikelihood(
                jsparse.BCOO((ll_values, indices), shape=(len(fragments), n_reads))
            )
            paired_lls = FragmentReadErrorLikelihood(
                jsparse.BCOO((paired_ll_values, paired_indices), shape=(len(fragment_pairs), n_read_pairs))
            )
            all_matrices.append(TimeSliceLikelihoods(lls, paired_lls))

        return TimeSeriesLikelihoods(all_matrices, fragments, fragment_pairs)

    def process_time_slice(
            self,
            time_slice: TimeSliceReads,
            fragments: FragmentSpace,
            fragment_pairs: FragmentPairSpace
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int]:
        """
        Parse all the alignments, of all sources in the time slice.
        Includes a special case if the source specifies mate pairs; see the implementation of
        handle_paired_reads() method.
        """
        indices: List[List[int]] = []
        ll_values: List[float] = []
        paired_indices: List[List[int]] = []
        paired_ll_values: List[float] = []
        read_set = UniqueReadOrdering()  # this is specific to the read time slice.
        paired_read_set = UniquePairedReadOrdering()  # ditto.

        for src in time_slice.sources:
            if isinstance(src, SampleReadSourceSingle):
                self.handle_singular_read_likelihoods(time_slice, src,
                                                      fragments, read_set,
                                                      indices, ll_values)
            elif isinstance(src, SampleReadSourcePaired):
                self.handle_paired_read_likelihoods(time_slice, src,
                                                    fragments, fragment_pairs,
                                                    read_set, paired_read_set,
                                                    indices, paired_indices,
                                                    ll_values, paired_ll_values)
        return (
            jnp.array(ll_values, dtype=self.dtype), jnp.array(paired_ll_values, dtype=self.dtype),
            jnp.array(indices), jnp.array(paired_indices),
            len(read_set), len(paired_read_set)
        )

    def get_alignments_singular(self,
                                read_src: SampleReadSourceSingle,
                                time_slice: TimeSliceReads) -> Iterator[SequenceReadPairwiseAlignment]:
        """
        A wrapper for performing alignments of single fastq files.
        """
        if read_src.read_type == ReadType.SINGLE_END:
            yield from self.alignment_wrapper.align_single_end(
                read_src.path, read_src.quality_format, time_slice.get_read
            )
        elif read_src.read_type == ReadType.PAIRED_END_1:
            yield from self.alignment_wrapper.align_pe_forward(
                read_src.path, read_src.quality_format, time_slice.get_read
            )
        elif read_src.read_type == ReadType.PAIRED_END_2:
            yield from self.alignment_wrapper.align_pe_reverse(
                read_src.path, read_src.quality_format, time_slice.get_read
            )
        else:
            raise ValueError(f"Don't know how to handle a source with singular file of type `{read_src.read_type}`")

    def get_alignments_paired(self,
                              read_src: SampleReadSourcePaired,
                              time_slice: TimeSliceReads) -> Iterator[SequenceReadPairwiseAlignment]:
        """
        A wrapper for performing alignments of paired-end fastq files;
        A quirk here is that we don't use aligners' built-in paired-end modes,
        instead (in order to maximize the # of fragment hits), we align each mate pair file as
        in single-ended mode (but using paired-end indel profiles).
        """
        yield from self.alignment_wrapper.align_pe_forward(
            read_src.path_fwd, read_src.quality_format, time_slice.get_read
        )
        yield from self.alignment_wrapper.align_pe_reverse(
            read_src.path_rev, read_src.quality_format, time_slice.get_read
        )

    def handle_singular_read_likelihoods(
            self,
            time_slice: TimeSliceReads,
            read_src: SampleReadSourceSingle,
            fragments: FragmentSpace,
            read_set: UniqueReadOrdering,
            indices: List[List[int]],
            ll_values: List[float]
    ):
        """
        Parse each alignment, and extract the marker fragment hit.
        """
        for read, _, frag, error_ll in self.parse_alignments(
                self.get_alignments_singular(read_src, time_slice),
                fragments
        ):
            frag_idx = frag.index
            read_idx = read_set.get_index_of(read)
            indices.append([frag_idx, read_idx])
            ll_values.append(error_ll)

    def handle_paired_read_likelihoods(
            self,
            time_slice: TimeSliceReads,
            read_src: SampleReadSourcePaired,
            fragments: FragmentSpace, fragment_pairs: FragmentPairSpace,
            read_set: UniqueReadOrdering, paired_read_set: UniquePairedReadOrdering,
            indices: List[List[int]], paired_indices: List[List[int]],
            ll_values: List[float], paired_ll_values: List[float]
    ):
        """
        Same as handle_singular_reads for most reads.
        If a mate pair jointly hits some collection of markers, then those fragments are paired and recorded.
        (All other alignments hitting non-identical markers are discarded; similar to taking the intersection
        of a pair of read mapping hits)
        """
        """
        The logic here is simple but long. Read through it carefully!
        """
        # SAM files are flat (one alignment per line), so we need to do some extra work to tally up the hit markers.
        # Use pandas to assist without too much performance impact.
        df_entries = []

        # First, handle the forward reads and create a dataframe of all alignment hits.
        for fwd_read, marker_gene, frag, error_ll in self.parse_alignments(
                self.alignment_wrapper.align_pe_forward(
                    read_src.path_fwd, read_src.quality_format, time_slice.get_read
                ),
                fragments
        ):
            strain = self.db.get_strains_with_marker(marker_gene)[0]
            df_entries.append({
                'Read': fwd_read.id,
                'Marker': marker_gene.id,
                'Strain': strain.id,
                'Frag': frag.index,
                'LL': error_ll
            })
        fwd_hit_df = pd.DataFrame(df_entries)
        del df_entries
        logger.debug("# Forward-read alignments = {}".format(fwd_hit_df.shape[0]))

        # Next, handle the reverse reads, do the same thing as above.
        df_entries = []
        for rev_read, marker_gene, frag, error_ll in self.parse_alignments(
                self.alignment_wrapper.align_pe_reverse(
                    read_src.path_rev, read_src.quality_format, time_slice.get_read
                ),
                fragments
        ):
            if rev_read.has_mate_pair:
                mate_pair_id = rev_read.mate_pair.id
            else:
                mate_pair_id = None
            strain = self.db.get_strains_with_marker(marker_gene)[0]
            df_entries.append({
                'Read': rev_read.id,
                'MatePair': mate_pair_id,
                'Marker': marker_gene.id,
                'Strain': strain.id,
                'Frag': frag.index,
                'LL': error_ll
            })
        rev_hit_df = pd.DataFrame(df_entries)
        del df_entries
        logger.debug("# Reverse-read alignments = {}".format(rev_hit_df.shape[0]))

        # Handle mate paired reads that hit the same marker (inner join the dataframes)
        fwd_to_remove = set()
        rev_to_remove = set()
        merged = fwd_hit_df.merge(
            rev_hit_df,
            left_on=['Read', 'Strain'],
            right_on=['MatePair', 'Strain'],
            how='inner',
            suffixes=('_FWD', '_REV')
        )
        logger.debug(
            "# Merged alignments = {} (ratio_fwd = {}, ratio_rev = {})".format(
                merged.shape[0],
                merged.shape[0] / fwd_hit_df.shape[0],
                merged.shape[0] / rev_hit_df.shape[0]
            )
        )
        for (fwd_read_id, marker_gene), row in merged.iterrows():
            fwd_read = time_slice.get_read(fwd_read_id)
            assert isinstance(fwd_read, PairedEndRead)
            assert fwd_read.has_mate_pair
            rev_read = fwd_read.mate_pair
            fwd_frag = fragments.get_fragment_by_index(row['Frag_FWD'])
            rev_frag = fragments.get_fragment_by_index(row['Frag_REV'])
            ll_err_total = row['LL_FWD'] + row['LL_REV']

            # paired indices. These will be stored in a separate matrix.
            paired_read_idx = paired_read_set.get_index_of((fwd_read, rev_read))
            paired_frag_idx = fragment_pairs.get_index(fwd_frag, rev_frag)
            paired_indices.append([paired_frag_idx, paired_read_idx])
            paired_ll_values.append(ll_err_total)

            # Remove these handled reads; this is akin to taking the intersection of mapping genomes.
            # Note that not all mate pairs will map properly, since markers are far sparser than the entire genome.
            fwd_to_remove.add(fwd_read_id)
            rev_to_remove.add(rev_read.id)
        del merged

        # Handle reads that don't have mate pairs, or mate pairs that don't share marker hits.
        logger.debug("Resolved {} reads with paired-end information.".format(len(fwd_to_remove)))
        fwd_hit_df = fwd_hit_df.loc[~fwd_hit_df['Read'].isin(fwd_to_remove)]
        rev_hit_df = rev_hit_df.loc[~rev_hit_df['Read'].isin(rev_to_remove)]
        for _, row in itertools.chain(fwd_hit_df.iterrows(), rev_hit_df.iterrows()):
            read_idx = read_set.get_index_from_key(row['Read'])
            frag_idx = row['Frag']
            indices.append([frag_idx, read_idx])
            ll_values.append(row['LL'])

    def parse_alignments(
            self,
            alns: Iterator[SequenceReadPairwiseAlignment],
            fragments: FragmentSpace
    ) -> Tuple[SequenceRead, Marker, Fragment, float]:
        """
        Parse the provided alignment objects into the necessar objects for calculating a
        read-fragment likelihood matrix.
        """
        # we can afford to clear this in between sources.
        included_hits = set()

        # parse each alignment, keeping track of duplicates along the way.
        for aln in alns:
            frag = fragments.add_seq(aln.marker_frag)  # this checks for reverse complements.
            pair_identifier: Tuple[str, int] = (aln.read.id, frag.index)
            if pair_identifier in included_hits:
                continue  # this is a duplicate alignment, so skip.
            else:
                included_hits.add(pair_identifier)

            error_ll = self.read_frag_ll(
                frag.seq.bytes(),
                aln.read,
                aln.read_insertion_locs(),
                aln.marker_deletion_locs(),
                aln.reverse_complemented,
                aln.soft_clip_start + aln.hard_clip_start,
                aln.soft_clip_end + aln.hard_clip_end
            )

            if error_ll > self.ll_threshold:
                yield aln.read, aln.marker, frag, error_ll

    def read_frag_ll(self,
                     frag_seq: cnp.ndarray,
                     read: SequenceRead,
                     insertions: cnp.ndarray,
                     deletions: cnp.ndarray,
                     reverse_complemented: bool,
                     start_clip: int,
                     end_clip: int):
        """
        Invoke the underlying phred/indel error model.
        the -np.log(2) is there due to a 0.5 chance of forward/reverse (rev_comp).
        This is an approximation of the dense version, assuming that either p_forward or p_reverse is
        approximately zero given the actual alignment.
        """
        forward_ll = self.error_model.compute_log_likelihood(
            frag_seq, read,
            read_reverse_complemented=reverse_complemented,
            insertions=insertions,
            deletions=deletions,
            read_start_clip=start_clip,
            read_end_clip=end_clip
        )
        return forward_ll - cnp.log(2)

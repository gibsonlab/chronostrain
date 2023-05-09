from typing import List, Dict, Tuple, Set, Iterator

import jax.numpy as np
import jax.experimental.sparse as jsparse

from chronostrain.database import StrainDatabase
from chronostrain.model import Fragment, SequenceRead
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.filesystem import convert_size
from chronostrain.model.io import TimeSeriesReads
from chronostrain.model.generative import GenerativeModel

from chronostrain.util.math import save_sparse_matrix, load_sparse_matrix
from .alignments import CachedReadMultipleAlignments, CachedReadPairwiseAlignments
from .cache import ReadsPopulationCache

from chronostrain.logging import create_logger

logger = create_logger(__name__)


# noinspection PyPep8Naming
class SparseDataLikelihoods:
    def __init__(
            self,
            model: GenerativeModel,
            data: TimeSeriesReads,
            db: StrainDatabase,
            read_likelihood_lower_bound: float = 1e-30,
            num_cores: int = 1
    ):
        self.model = model
        self.data = data
        self.db = db
        self.read_likelihood_lower_bound = read_likelihood_lower_bound
        self.num_cores = num_cores

        log_likelihoods_tensors = SparseLogLikelihoodComputer(
            self.model, self.data, self.db, self.num_cores
        ).compute_likelihood_tensors()

        self.matrices: List[jsparse.BCOO] = [
            ll_tensor for ll_tensor in log_likelihoods_tensors
        ]
        self.supported_frags: List[np.ndarray] = []

        # Pre-compute supported fragments for each timepoint t.
        for t_idx, t in enumerate(self.model.times):
            F, R = self.matrices[t_idx].shape
            row_support = np.unique(
                self.matrices[t_idx].indices[:, 0],
                return_inverse=False, return_counts=False  # np.unique sorts by default
            )
            _F = len(row_support)
            logger.debug("(t = {}) # of supported fragments: {} out of {} ({:.2e}) ({} reads)".format(
                t, _F, F, _F / F, len(data[t_idx])
            ))
            self.supported_frags.append(row_support)

    def reduce_supported_fragments(self, x: jsparse.BCOO, t_idx: int, fragment_dim: int) -> jsparse.BCOO:
        import numpy as cnp
        F_REDUCED = len(self.supported_frags[t_idx])
        if fragment_dim == 0:
            shape = (F_REDUCED, x.shape[1])
        elif fragment_dim == 1:
            shape = (x.shape[0], F_REDUCED)
        else:
            raise ValueError("Unexpected fragment_dim argument. Must be 0 or 1.")

        indices = cnp.array(x.indices)
        indices[:, fragment_dim] = np.digitize(
            x.indices[:, fragment_dim],
            bins=self.supported_frags[t_idx],
            right=True
        )
        indices = np.array(indices, dtype=int)

        return jsparse.BCOO((x.data, indices), shape=shape)


class SparseLogLikelihoodComputer:
    def __init__(self,
                 model: GenerativeModel,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 num_cores: int = 1):
        self.model = model
        self.reads = reads
        self._bwa_index_finished = False
        self.num_cores = num_cores

        # ==== Alignments of reads to the database reference markers.
        self.pairwise_reference_alignments = CachedReadPairwiseAlignments(reads, db)

        # ==== Multiple alignment of all reads to a single reference marker at a time.
        self.multiple_alignments = CachedReadMultipleAlignments(reads, db)

        # noinspection PyTypeChecker
        self._multi_align_instances: List[MarkerMultipleFragmentAlignment] = None  # lazy loading

        # ==== Cache.
        self.cache = ReadsPopulationCache(reads, model.bacteria_pop)

    def read_frag_ll(self,
                     frag_seq: np.ndarray,
                     read: SequenceRead,
                     insertions: np.ndarray,
                     deletions: np.ndarray,
                     reverse_complemented: bool,
                     start_clip: int,
                     end_clip: int):
        """
        the -np.log(2) is there due to a 0.5 chance of forward/reverse (rev_comp).
        This is an approximation of the dense version, assuming that either p_forward or p_reverse is
        approximately zero given the actual alignment.
        """
        forward_ll = self.model.error_model.compute_log_likelihood(
            frag_seq, read,
            read_reverse_complemented=reverse_complemented,
            insertions=insertions,
            deletions=deletions,
            read_start_clip=start_clip,
            read_end_clip=end_clip
        )
        return forward_ll - np.log(2)

    def _compute_read_frag_alignments_pairwise(self, t_idx: int) -> Iterator[Tuple[SequenceRead, str, float]]:
        """
        Iterate through the timepoint's SamHandler instances (if the reads of t_idx came from more than one path).
        Each of these SamHandlers provides alignment information.
        :param t_idx: the timepoint index to use.
        :return: An iterator over triples (read_id, frag_seq, error_ll).
        """

        # :return: A defaultdict representing the map
        #     (Read ID) -> {Fragments that the read aligns to}, {Frag->Read log likelihood}
        # """
        # read_to_frag_likelihoods: Dict[str, List[Tuple[Fragment, float]]] = defaultdict(list)

        """
        Does NOT handle alignment of indels to indels! (e.g. for marker variant construction). By definition, 
        that requires multiple alignment.
        """
        ll_threshold = -500
        for aln in self.pairwise_reference_alignments.alignments_by_timepoint(t_idx):
            included_pairs: Set[str] = set()

            if self.model.bacteria_pop.contains_marker(aln.marker):
                pair_identifier = f"{aln.read.id}->{aln.marker_frag.nucleotides()}"
                if pair_identifier in included_pairs:
                    continue
                else:
                    included_pairs.add(pair_identifier)

                error_ll = self.read_frag_ll(
                    aln.marker_frag.bytes(),
                    aln.read,
                    aln.read_insertion_locs(),
                    aln.marker_deletion_locs(),
                    aln.reverse_complemented,
                    aln.soft_clip_start + aln.hard_clip_start,
                    aln.soft_clip_end + aln.hard_clip_end
                )

                if error_ll > ll_threshold:
                    yield aln.read, aln.marker_frag, error_ll

    def _compute_read_frag_alignments_multiple(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, float]]]:
        """
        Performs a multiple alignment to determine the fragments, for either forward or reverse alignments.
        Should be safer to indels than pairwise.

        :param t_idx: the timepoint index to use.
        :return: A defaultdict representing the map (Read ID) -> {Fragments that the read aligns to}
        """
        raise NotImplementedError()

        # read_to_frag_likelihoods: Dict[str, List[Tuple[Fragment, float]]] = defaultdict(list)
        #
        # logger.debug(f"(t = {t_idx}) Retrieving multiple alignments.")
        # if self._multi_align_instances is None:
        #     self._multi_align_instances = list(self.multiple_alignments.get_alignments(num_cores=self.num_cores))
        #
        # time_slice = self.reads[t_idx]
        # included_pairs: Set[str] = set()
        # ll_threshold = -500
        #
        # """
        # Helper function (Given subseq/read pair (and other relevant information), compute likelihood and insert into matrix.
        # """
        # def add_subseq_likelihood(subseq, read, insertions, deletions, revcomp, start_clip: int, end_clip: int):
        #     frag = self.model.fragments.get_fragment(subseq)
        #
        #     pair_identifier = f"{read.id}->{frag.index}"
        #     if pair_identifier in included_pairs:
        #         return
        #     else:
        #         included_pairs.add(pair_identifier)
        #
        #     ll = self.read_frag_ll(
        #         frag,
        #         read,
        #         insertions, deletions,
        #         reverse_complemented=revcomp,
        #         start_clip=start_clip,
        #         end_clip=end_clip
        #     )
        #
        #     if ll < ll_threshold:
        #         return
        #     read_to_frag_likelihoods[read.id].append((frag, ll))
        #
        # """
        # Main loop
        # """
        # for multi_align in self._multi_align_instances:
        #     logger.debug(f"[{multi_align.canonical_marker.name}] Processing alignment of reads "
        #                  f"({len(multi_align.forward_read_index_map)} forward, "
        #                  f"{len(multi_align.reverse_read_index_map)} reverse) "
        #                  f"into likelihoods.")
        #
        #     for frag_entry in multi_align.all_mapped_fragments():
        #         marker, read, subseq, insertions, deletions, start_clip, end_clip, revcomp = frag_entry
        #         add_subseq_likelihood(subseq, read, insertions, deletions, revcomp, start_clip, end_clip)
        #
        #     # Next, take care of the variant markers (if applicable).
        #     for variant in self.marker_variants_of(multi_align.canonical_marker):
        #         for read in time_slice:
        #             for subseq, revcomp, insertions, deletions, start_clip, end_clip in variant.subseq_from_read(read):
        #                 add_subseq_likelihood(subseq, read, insertions, deletions, revcomp, start_clip, end_clip)
        #
        # return read_to_frag_likelihoods

    def create_sparse_matrix(self, t_idx: int) -> jsparse.BCOO:
        """
        For the specified time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param t_idx: The time point index to run this function on.
        :returns: A sparse representation of read indices, frag indices and the corresponding log-likelihoods,
        log P(read | frag).
        """
        # Perform alignment for approximate fine-grained search.
        # read_to_frag_likelihoods: Dict[str, List[Tuple[Fragment, float]]] = self._compute_read_frag_alignments(t_idx)

        indices: List[List[int]] = []
        log_likelihood_values: List[float] = []

        for read, frag_seq, error_ll in self._compute_read_frag_alignments_pairwise(t_idx):
            indices.append([self.model.fragments.get_fragment_index(frag_seq), read.index])
            log_likelihood_values.append(error_ll)

        return jsparse.BCOO(
            (
                np.array(log_likelihood_values),
                np.array(indices)
            ),
            shape=(len(self.model.fragments), len(self.reads[t_idx])),
        )

    def compute_likelihood_tensors(self) -> List[jsparse.BCOO]:
        """
        Invokes create_sparse_matrix by passing it through the cache.
        :return:
        """
        logger.debug("Computing read-fragment likelihoods...")

        def _matrix_load_callback(matrix: jsparse.BCOO):
            _, counts_per_read = np.unique(matrix.indices[1], return_counts=True)
            logger.debug(
                "Read-likelihood matrix (size {r} x {c}) has {nz} nonzero entries. "
                "(~{meanct:.2f} hits per read, physical({dev})={phys})".format(
                    r=matrix.shape[0],
                    c=matrix.shape[1],
                    nz=len(matrix.data),
                    meanct=counts_per_read.mean().item(),
                    dev="todo",
                    phys="todo"
                )
            )

        return [
            self.cache.call(
                "sparse_log_likelihoods_{}.npz".format(t_idx),
                self.create_sparse_matrix,
                [],
                {"t_idx": t_idx},
                save_sparse_matrix,
                load_sparse_matrix,
                _matrix_load_callback
            )
            for t_idx in range(self.model.num_times())
        ]

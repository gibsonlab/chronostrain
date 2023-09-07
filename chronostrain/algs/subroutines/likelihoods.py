from typing import List, Dict, Tuple, Set, Iterator

import jax.numpy as np
import jax.experimental.sparse as jsparse

from chronostrain.database import StrainDatabase
from chronostrain.model import Fragment, SequenceRead
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.model.io import TimeSeriesReads
from chronostrain.model.generative import GenerativeModel

from chronostrain.util.math import save_sparse_matrix, load_sparse_matrix
from .alignments import CachedReadPairwiseAlignments
from .cache import ReadsPopulationCache

from chronostrain.logging import create_logger
from ...util.sequences import Sequence

logger = create_logger(__name__)


# noinspection PyPep8Naming
class SparseDataLikelihoods:
    def __init__(
            self,
            model: GenerativeModel,
            data: TimeSeriesReads,
            db: StrainDatabase,
            read_likelihood_lower_bound: float = 1e-30,
            num_cores: int = 1,
            dtype='bfloat16'
    ):
        self.model = model
        self.data = data
        self.db = db
        self.read_likelihood_lower_bound = read_likelihood_lower_bound
        self.num_cores = num_cores

        log_likelihoods_tensors = SparseLogLikelihoodComputer(
            self.model, self.data, self.db, self.num_cores,
            dtype=dtype
        ).compute_likelihood_tensors()

        self.matrices: List[jsparse.BCOO] = [
            ll_tensor for ll_tensor in log_likelihoods_tensors
        ]


class SparseLogLikelihoodComputer:
    def __init__(self,
                 model: GenerativeModel,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 num_cores: int = 1,
                 dtype='bfloat16'):
        self.model = model
        self.reads = reads
        self._bwa_index_finished = False
        self.num_cores = num_cores
        self.dtype = dtype

        # ==== Alignments of reads to the database reference markers.
        self.pairwise_reference_alignments = CachedReadPairwiseAlignments(reads, db, num_cores=self.num_cores)

        # noinspection PyTypeChecker
        self._multi_align_instances: List[MarkerMultipleFragmentAlignment] = None  # lazy loading

        # ==== Cache.
        self.cache = ReadsPopulationCache(reads, db)

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

    def _compute_read_frag_alignments_pairwise(self, t_idx: int) -> Iterator[Tuple[SequenceRead, Sequence, float]]:
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
        ll_threshold = -100
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

        if len(indices) > 0:
            return jsparse.BCOO(
                (
                    np.array(log_likelihood_values, dtype=self.dtype),
                    np.array(indices, dtype=int)
                ),
                shape=(len(self.model.fragments), len(self.reads[t_idx])),
            )
        else:
            return jsparse.BCOO(
                (
                    np.empty(shape=(0,), dtype=self.dtype),
                    np.empty(shape=(0, 2), dtype=int),
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
            _, counts_per_read = np.unique(matrix.indices[:, 1], return_counts=True)
            logger.debug(
                "Read-likelihood matrix (size {r} x {c}) has {nz} nonzero entries. "
                "(~{meanct:.2f}±{stdct:.2f} hits per read)".format(
                    r=matrix.shape[0],
                    c=matrix.shape[1],
                    nz=len(matrix.data),
                    meanct=counts_per_read.mean().item(),
                    stdct=counts_per_read.std().item()
                )
            )

        self.cache.create_subdir('log_likelihoods')
        return [
            self.cache.call(
                "log_likelihoods/sparse_log_likelihoods_{}.{}.npz".format(t_idx, self.dtype),
                self.create_sparse_matrix,
                [],
                {"t_idx": t_idx},
                save_sparse_matrix,
                load_sparse_matrix,
                _matrix_load_callback
            )
            for t_idx in range(self.model.num_times())
        ]

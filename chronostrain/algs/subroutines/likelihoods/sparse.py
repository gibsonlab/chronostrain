import torch
from typing import List, Dict, Set, Iterator, Tuple
from collections import defaultdict
import numpy as np

from joblib import Parallel, delayed

from chronostrain.algs.subroutines.alignments import CachedReadPairwiseAlignments, CachedReadMultipleAlignments
from chronostrain.database import StrainDatabase
from chronostrain.model import Fragment, Marker, SequenceRead
from chronostrain.util.alignments.multiple import MarkerMultipleAlignment
from chronostrain.util.sequences import SeqType
from chronostrain.util.sparse.sparse_tensor import SparseMatrix
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from .base import DataLikelihoods, AbstractLogLikelihoodComputer
from .likelihood_cache import LikelihoodMatrixCache

from chronostrain.config.logging import create_logger
from chronostrain.algs.variants import MarkerVariant

logger = create_logger(__name__)


# noinspection PyPep8Naming
class SparseDataLikelihoods(DataLikelihoods):
    def __init__(
            self,
            model: GenerativeModel,
            data: TimeSeriesReads,
            db: StrainDatabase,
            read_likelihood_lower_bound: float = 1e-30
    ):
        self.db = db
        super().__init__(model, data, read_likelihood_lower_bound=read_likelihood_lower_bound)
        self.supported_frags = []

        # Delete empty rows.
        for t_idx in range(self.model.num_times()):
            F = self.matrices[t_idx].size()[0]
            row_support = self.matrices[t_idx].indices[0, :].unique(
                sorted=True, return_inverse=False, return_counts=False
            )
            _F = len(row_support)

            _support_indices = torch.tensor([
                [i for i in range(len(row_support))],
                [row_support[i] for i in range(_F)]
            ], dtype=torch.long, device=cfg.torch_cfg.device)

            projector = SparseMatrix(
                indices=_support_indices,
                values=torch.ones(_support_indices.size()[1],
                                  device=cfg.torch_cfg.device,
                                  dtype=cfg.torch_cfg.default_dtype),
                dims=(_F, F)
            )

            self.matrices[t_idx] = projector.sparse_mul(self.matrices[t_idx])
            self.supported_frags.append(row_support)

    def _likelihood_computer(self) -> AbstractLogLikelihoodComputer:
        return SparseLogLikelihoodComputer(self.model, self.data, self.db)


class SparseLogLikelihoodComputer(AbstractLogLikelihoodComputer):
    def __init__(self,
                 model: GenerativeModel,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 alignment_mode: str = "multiple"):
        super().__init__(model, reads)
        self._bwa_index_finished = False

        # ==== Alignments of reads to the database reference markers.
        self.pairwise_reference_alignments = CachedReadPairwiseAlignments(reads, db)

        # ==== Multiple alignment of all reads to a single reference marker at a time.
        self.multiple_alignments = CachedReadMultipleAlignments(reads, db)

        # noinspection PyTypeChecker
        self._multi_align_instances: List[MarkerMultipleAlignment] = None  # lazy loading

        # ==== Cache.
        self.cache = LikelihoodMatrixCache(reads, model.bacteria_pop)

        self.markers_present: Set[Marker] = set()
        self.variants_present: Dict[Marker, List[MarkerVariant]] = {
            marker: []
            for marker in db.all_markers()
        }

        for marker in model.bacteria_pop.markers_iterator():
            if isinstance(marker, MarkerVariant):
                self.variants_present[marker.base_marker].append(marker)
            else:
                self.markers_present.add(marker)

        self.alignment_mode = alignment_mode

    def marker_variants_of(self, marker: Marker) -> Iterator[MarkerVariant]:
        yield from self.variants_present[marker]

    def marker_isin_pop(self, marker: Marker) -> bool:
        return marker in self.markers_present

    def _compute_read_frag_alignments(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, float]]]:
        if self.alignment_mode == "pairwise":
            return self._compute_read_frag_alignments_pairwise(t_idx)
        elif self.alignment_mode == "multiple":
            return self._compute_read_frag_alignments_multiple(t_idx)
        else:
            raise ValueError(f"Unexpected alignment_mode argument `{self.alignment_mode}`.")

    def read_frag_ll(self,
                     frag: Fragment,
                     read: SequenceRead,
                     insertions: np.ndarray,
                     deletions: np.ndarray,
                     reverse_complemented: bool):
        """
        the -np.log(2) is there due to a 0.5 chance of forward/reverse (rev_comp).
        This is an approximation of the dense version, assuming that either p_forward or p_reverse is
        approximately zero given the actual alignment.
        """
        forward_ll = self.model.error_model.compute_log_likelihood(
            frag, read, read_reverse_complemented=reverse_complemented, insertions=insertions, deletions=deletions
        )
        return forward_ll - np.log(2)

    def _compute_read_frag_alignments_pairwise(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, float]]]:
        """
        Iterate through the timepoint's SamHandler instances (if the reads of t_idx came from more than one path).
        Each of these SamHandlers provides alignment information.
        :param t_idx: the timepoint index to use.
        :return: A defaultdict representing the map
            (Read ID) -> {Fragments that the read aligns to}, {Frag->Read log likelihood}
        """
        read_to_fragments: Dict[str, List[Tuple[Fragment, float]]] = defaultdict(list)

        """
        Overall philosophy of this method: Keep things as simple as possible!
        Assume that indels have either been taken care of, either by passing in:
            1) appropriate MarkerVariant instances into the population, or
            2) include a complete reference cohort of Markers into the reference db 
            (but this is infeasible, as it requires knowing the ground truth on real data!)
        
        In particular, this means that we don't have to worry about indels.
        """
        for base_marker, alns in self.pairwise_reference_alignments.alignments_by_marker_and_timepoint(t_idx).items():
            for aln in alns:
                # First, add the likelihood for the fragment for the aligned base marker.
                if self.marker_isin_pop(base_marker):
                    marker_frag_seq: SeqType = aln.marker_frag
                    aln_insertion_locs = aln.read_insertion_locs()
                    if aln.reverse_complemented:
                        aln_insertion_locs = aln_insertion_locs[::-1]

                    try:
                        tgt_frag = self.model.fragments.get_fragment(marker_frag_seq)
                        read_ll = self.read_frag_ll(
                            frag=tgt_frag,
                            read=aln.read,
                            insertions=aln_insertion_locs,
                            deletions=aln.marker_deletion_locs(),
                            reverse_complemented=aln.reverse_complemented
                        )
                        read_to_fragments[aln.read.id].append((tgt_frag, read_ll))
                    except KeyError:
                        # Ignore these errors (see above note).
                        pass

                # Next, look up any variants of the base marker.
                for variant in self.marker_variants_of(base_marker):
                    variant_frag_seq, variant_read_insertions, variant_marker_deletions = variant.subseq_from_ref_alignment(aln)
                    if aln.read_start > 0 or aln.read_end < len(aln.read.seq) - 1:
                        # Read only partially maps to marker (usually edge effect).
                        logger.debug(
                            f"Discarding alignment of read {aln.read.id} (length={len(aln.read.seq)}) "
                            f"to marker {aln.marker.id}, alignment got clipped "
                            f"(read_start = {aln.read_start}, read_end = {aln.read_end})"
                        )
                        continue

                    variant_frag = self.model.fragments.get_fragment(variant_frag_seq)
                    if aln.reverse_complemented:
                        variant_read_insertions = variant_read_insertions[::-1]
                    try:
                        read_ll = self.read_frag_ll(
                            read=aln.read,
                            frag=variant_frag,
                            insertions=variant_read_insertions,
                            deletions=variant_marker_deletions,
                            reverse_complemented=aln.reverse_complemented
                        )
                        read_to_fragments[aln.read.id].append((variant_frag, read_ll))
                    except KeyError:
                        logger.debug(
                            f"Line {aln.sam_line_no} points to Read `{aln.read.id}`, "
                            f"but encountered KeyError. (Sam = {aln.sam_path})"
                        )
                        raise
        return read_to_fragments

    def _compute_read_frag_alignments_multiple(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, float]]]:
        """
        Performs a multiple alignment to determine the fragments, for either forward or reverse alignments.
        Should be safer to indels than

        :param t_idx: the timepoint index to use.
        :return: A defaultdict representing the map (Read ID) -> {Fragments that the read aligns to}
        """
        read_to_frag_likelihoods: Dict[str, List[Tuple[Fragment, float]]] = defaultdict(list)

        if self._multi_align_instances is None:
            self._multi_align_instances = list(self.multiple_alignments.get_alignments())

        time_slice = self.reads[t_idx]
        for multi_align in self._multi_align_instances:
            # First, take care of the base markers (if applicable).
            if self.marker_isin_pop(multi_align.marker):
                for reverse in [False, True]:
                    for read_id in multi_align.read_ids(reverse=reverse):
                        read = time_slice.get_read(read_id)
                        subseq, insertions, deletions = multi_align.get_aligned_reference_region(
                            read_id, reverse=reverse
                        )

                        # Key point: insertions is ordered with respect to the alignment
                        # (which is reversed if reverse=True).
                        if reverse:
                            insertions = insertions[::-1]

                        frag = self.model.fragments.get_fragment(subseq)

                        ll = self.read_frag_ll(frag, read, insertions, deletions, reverse_complemented=reverse)
                        read_to_frag_likelihoods[read_id].append((frag, ll))

            # Next, take care of the variant markers (if applicable).
            ### TODO -- test run without variants before proceeding!

        return read_to_frag_likelihoods

    def create_sparse_matrix(self, t_idx: int) -> SparseMatrix:
        """
        For the specified time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param t_idx: The time point index to run this function on.
        :returns: A sparse representation of read indices, frag indices and the corresponding log-likelihoods,
        log P(read | frag).
        """
        # Perform alignment for approximate fine-grained search.
        read_to_frag_likelihoods: Dict[str, List[Tuple[Fragment, float]]] = self._compute_read_frag_alignments(t_idx)

        read_indices: List[int] = []
        frag_indices: List[int] = []
        log_likelihood_values: List[float] = []
        for read_idx, read in enumerate(self.reads[t_idx]):
            for frag, log_likelihood in read_to_frag_likelihoods[read.id]:
                read_indices.append(read_idx)
                frag_indices.append(frag.index)
                log_likelihood_values.append(log_likelihood)

        return SparseMatrix(
            indices=torch.tensor(
                [frag_indices, read_indices],
                device=cfg.torch_cfg.device,
                dtype=torch.long
            ),
            values=torch.tensor(
                log_likelihood_values,
                device=cfg.torch_cfg.device,
                dtype=cfg.torch_cfg.default_dtype
            ),
            dims=(self.model.fragments.size(), len(self.reads[t_idx]))
        )

    def compute_likelihood_tensors(self) -> List[SparseMatrix]:
        logger.debug("Computing read-fragment likelihoods...")

        # Save each sparse tensor as a tuple of indices/values/shape into a compressed numpy file (.npz).
        def save_(path, sparse_matrix: SparseMatrix):
            np.savez(
                path,
                sparse_indices=sparse_matrix.indices.cpu().numpy(),
                sparse_values=sparse_matrix.values.cpu().numpy(),
                matrix_shape=np.array([
                    sparse_matrix.rows,
                    sparse_matrix.columns
                ])
            )

        def load_(path) -> SparseMatrix:
            data = np.load(path)
            size = data["matrix_shape"]
            return SparseMatrix(
                indices=torch.tensor(
                    data['sparse_indices'],
                    device=cfg.torch_cfg.device,
                    dtype=torch.long
                ),
                values=torch.tensor(
                    data['sparse_values'],
                    device=cfg.torch_cfg.device,
                    dtype=cfg.torch_cfg.default_dtype
                ),
                dims=(size[0], size[1])
            )

        def callback_(matrix: SparseMatrix):
            _, counts_per_read = torch.unique(matrix.indices[1], sorted=False, return_inverse=False, return_counts=True)
            logger.debug(
                "Read-likelihood matrix (size {r} x {c}) has {nz} nonzero entries. "
                "(~{meanct:.2f} hits per read, density={dens:.1e})".format(
                    r=matrix.size()[0],
                    c=matrix.size()[1],
                    nz=len(matrix.values),
                    meanct=counts_per_read.float().mean(),
                    dens=matrix.density()
                )
            )

        jobs = [
            {
                "filename": "sparse_log_likelihoods_{}.npz".format(t_idx),
                "fn": lambda t: self.create_sparse_matrix(t),
                "args": [],
                "kwargs": {"t": t_idx},
                "save": save_,
                "load": load_,
                "success_callback": callback_
            }
            for t_idx in range(self.model.num_times())
        ]

        parallel = (cfg.model_cfg.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))

            return Parallel(n_jobs=cfg.model_cfg.num_cores)(
                delayed(self.cache.call)(cache_kwargs_t)
                for cache_kwargs_t in jobs
            )
        else:
            return [self.cache.call(**cache_kwargs_t) for cache_kwargs_t in jobs]

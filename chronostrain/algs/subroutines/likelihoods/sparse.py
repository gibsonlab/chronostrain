from pathlib import Path
from typing import List, Dict, Iterator, Tuple, Set
from collections import defaultdict

from joblib import Parallel, delayed

from chronostrain.database import StrainDatabase
from chronostrain.model import Fragment, Marker, SequenceRead, AbstractMarkerVariant
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.filesystem import convert_size
from chronostrain.util.math import *
from chronostrain.util.sparse import SparseMatrix, ColumnSectionedSparseMatrix
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from .base import DataLikelihoods, AbstractLogLikelihoodComputer
from ..alignments import CachedReadMultipleAlignments, CachedReadPairwiseAlignments
from ..cache import ReadsPopulationCache

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


# noinspection PyPep8Naming
class SparseDataLikelihoods(DataLikelihoods):
    def __init__(
            self,
            model: GenerativeModel,
            data: TimeSeriesReads,
            db: StrainDatabase,
            read_likelihood_lower_bound: float = 1e-30,
            num_cores: int = 1
    ):
        self.db = db
        self.num_cores = num_cores
        super().__init__(model, data, read_likelihood_lower_bound=read_likelihood_lower_bound)
        self.supported_frags: List[torch.Tensor] = []
        self.projectors: List[ColumnSectionedSparseMatrix] = []

        # Delete empty rows (Fragments)
        for t_idx in range(self.model.num_times()):
            F = self.matrices[t_idx].size()[0]
            row_support = self.matrices[t_idx].indices[0, :].unique(
                sorted=True, return_inverse=False, return_counts=False
            )
            _F = len(row_support)
            logger.debug("(t = {}) # of supported fragments: {} out of {} ({:.2e})".format(
                t_idx, _F, F, _F / F
            ))

            _support_indices = torch.tensor([
                [i for i in range(len(row_support))],
                [row_support[i] for i in range(_F)]
            ], dtype=torch.long, device=cfg.torch_cfg.device)

            projector = ColumnSectionedSparseMatrix(
                indices=_support_indices,
                values=torch.ones(_support_indices.size()[1],
                                  device=cfg.torch_cfg.device,
                                  dtype=cfg.torch_cfg.default_dtype),
                dims=(_F, F)
            )

            self.matrices[t_idx] = ColumnSectionedSparseMatrix.from_sparse_matrix(
                projector.sparse_mul(self.matrices[t_idx])
            )  # list of (F' x R)
            self.projectors.append(projector)
            self.supported_frags.append(row_support)

    def likelihood_matrix(self, t_idx: int) -> ColumnSectionedSparseMatrix:
        return self.matrices[t_idx]

    def _likelihood_computer(self) -> AbstractLogLikelihoodComputer:
        return SparseLogLikelihoodComputer(self.model, self.data, self.db, self.num_cores)

    def conditional_likelihood(self, X: torch.Tensor, inf_fill: float = -100000) -> float:
        y = torch.softmax(X, dim=1)
        total_ll = 0.
        for t_idx in range(self.model.num_times()):
            projector_t = self.projectors[t_idx]
            if projector_t.rows == 0 and projector_t.columns > 0:
                log_likelihood_t = -1e10 * len(self.data[t_idx])
            else:
                log_likelihood_t = log_mm_exp(
                    y[t_idx].log().view(1, -1),  # (N x S)
                    log_spmm_exp(
                        ColumnSectionedSparseMatrix.from_sparse_matrix(self.likelihood_matrix(t_idx).t()),  # (R x F')
                        log_spspmm_exp(
                            projector_t,  # (F' x F)
                            self.model.fragment_frequencies_sparse  # (F x S)
                        ),  # (F' x S)
                    ).t()  # after transpose: (S x R)
                )

                log_likelihood_t[torch.isinf(log_likelihood_t)] = inf_fill
                log_likelihood_t = log_likelihood_t.sum()
            total_ll += log_likelihood_t
        return total_ll


class SparseLogLikelihoodComputer(AbstractLogLikelihoodComputer):
    def __init__(self,
                 model: GenerativeModel,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 num_cores: int = 1):
        super().__init__(model, reads)
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

        # ==== Marker variants present in population.
        self.variants_present: Dict[Marker, Set[AbstractMarkerVariant]] = {
            marker: set()
            for marker in db.all_canonical_markers()
        }

        for marker in model.bacteria_pop.markers_iterator():
            if isinstance(marker, AbstractMarkerVariant):
                self.variants_present[marker.base_marker].add(marker)

    def marker_variants_of(self, marker: Marker) -> Iterator[AbstractMarkerVariant]:
        yield from self.variants_present[marker]

    def _compute_read_frag_alignments(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, float]]]:
        # Note: this method used to have multiple implementations.
        # Now we provide a default one which uses multiple alignment.
        # See _compute_read_frag_alignments_pairwise for the defunct implementation.

        return self._compute_read_frag_alignments_multiple(t_idx)
        # return self._compute_read_frag_alignments_pairwise(t_idx)

    def read_frag_ll(self,
                     frag: Fragment,
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
            frag, read,
            read_reverse_complemented=reverse_complemented,
            insertions=insertions,
            deletions=deletions,
            read_start_clip=start_clip,
            read_end_clip=end_clip
        )
        return forward_ll - np.log(2)

    # def _compute_read_frag_alignments_pairwise(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, float]]]:
    #     """
    #     Iterate through the timepoint's SamHandler instances (if the reads of t_idx came from more than one path).
    #     Each of these SamHandlers provides alignment information.
    #     :param t_idx: the timepoint index to use.
    #     :return: A defaultdict representing the map
    #         (Read ID) -> {Fragments that the read aligns to}, {Frag->Read log likelihood}
    #     """
    #     read_to_fragments: Dict[str, List[Tuple[Fragment, float]]] = defaultdict(list)
    #
    #     """
    #     Overall philosophy of this method: Keep things as simple as possible!
    #     Assume that indels have either been taken care of, either by passing in:
    #         1) appropriate MarkerVariant instances into the population, or
    #         2) include a complete reference cohort of Markers into the reference db
    #         (but this is infeasible, as it requires knowing the ground truth on real data!)
    #
    #     In particular, this means that we don't have to worry about indels.
    #     """
    #     logger.warning("_compute_read_frag_alignments_pairwise: Treating hard clipped reads the same as "
    #                    "soft clipped reads. (Developer note: keep an eye on this)")
    #     for base_marker, alns in self.pairwise_reference_alignments.alignments_by_marker_and_timepoint(t_idx).items():
    #         for aln in alns:
    #             if aln.is_edge_mapped or aln.is_clipped:
    #                 logger.debug(f"Ignoring alignment of read {aln.read.id} to marker {aln.marker.id} "
    #                              f"({aln.sam_path.name}, Line {aln.sam_line_no}), which is edge-mapped.")
    #                 continue
    #             # First, add the likelihood for the fragment for the aligned base marker.
    #             if self.model.bacteria_pop.contains_marker(base_marker):
    #                 marker_frag_seq = aln.marker_frag
    #                 aln_insertion_locs = aln.read_insertion_locs()
    #                 if aln.reverse_complemented:
    #                     aln_insertion_locs = aln_insertion_locs[::-1]
    #
    #                 try:
    #                     tgt_frag = self.model.fragments.get_fragment(marker_frag_seq)
    #                     read_ll = self.read_frag_ll(
    #                         frag=tgt_frag,
    #                         read=aln.read,
    #                         insertions=aln_insertion_locs,
    #                         deletions=aln.marker_deletion_locs(),
    #                         reverse_complemented=aln.reverse_complemented,
    #                         start_clip=0,
    #                         end_clip=0
    #                     )
    #                     read_to_fragments[aln.read.id].append((tgt_frag, read_ll))
    #                 except KeyError:
    #                     # Ignore these errors (see above note).
    #                     pass
    #
    #             # Next, look up any variants of the base marker.
    #             for variant in self.marker_variants_of(base_marker):
    #                 v_marker_frag_seq, v_read_insertions, v_marker_deletions = variant.subseq_from_pairwise_aln(aln)
    #                 if aln.read_start > 0 or aln.read_end < len(aln.read.seq) - 1:
    #                     # Read only partially maps to marker (usually edge effect).
    #                     logger.debug(
    #                         f"Discarding alignment of read {aln.read.id} (length={len(aln.read.seq)}) "
    #                         f"to marker {aln.marker.id}, alignment got clipped "
    #                         f"(read_start = {aln.read_start}, read_end = {aln.read_end})"
    #                     )
    #                     continue
    #
    #                 variant_frag = self.model.fragments.get_fragment(v_marker_frag_seq)
    #                 if aln.reverse_complemented:
    #                     v_read_insertions = v_read_insertions[::-1]
    #                 try:
    #                     read_ll = self.read_frag_ll(
    #                         read=aln.read,
    #                         frag=variant_frag,
    #                         insertions=v_read_insertions,
    #                         deletions=v_marker_deletions,
    #                         reverse_complemented=aln.reverse_complemented,
    #                         start_clip=0,
    #                         end_clip=0
    #                     )
    #                     read_to_fragments[aln.read.id].append((variant_frag, read_ll))
    #                 except KeyError:
    #                     logger.debug(
    #                         f"Line {aln.sam_line_no} points to Read `{aln.read.id}`, "
    #                         f"but encountered KeyError. (Sam = {aln.sam_path})"
    #                     )
    #                     raise
    #     return read_to_fragments

    def _compute_read_frag_alignments_multiple(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, float]]]:
        """
        Performs a multiple alignment to determine the fragments, for either forward or reverse alignments.
        Should be safer to indels than pairwise.

        :param t_idx: the timepoint index to use.
        :return: A defaultdict representing the map (Read ID) -> {Fragments that the read aligns to}
        """
        read_to_frag_likelihoods: Dict[str, List[Tuple[Fragment, float]]] = defaultdict(list)

        logger.debug(f"(t = {t_idx}) Retrieving multiple alignments.")
        if self._multi_align_instances is None:
            self._multi_align_instances = list(self.multiple_alignments.get_alignments(num_cores=self.num_cores))

        time_slice = self.reads[t_idx]
        included_pairs: Set[str] = set()
        ll_threshold = -500

        """
        Helper function (Given subseq/read pair (and other relevant information), compute likelihood and insert into matrix.
        """
        def add_subseq_likelihood(subseq, read, insertions, deletions, revcomp, start_clip: int, end_clip: int):
            frag = self.model.fragments.get_fragment(subseq)

            pair_identifier = f"{read.id}->{frag.index}"
            if pair_identifier in included_pairs:
                return
            else:
                included_pairs.add(pair_identifier)

            ll = self.read_frag_ll(
                frag,
                read,
                insertions, deletions,
                reverse_complemented=revcomp,
                start_clip=start_clip,
                end_clip=end_clip
            )

            if ll < ll_threshold:
                return
            read_to_frag_likelihoods[read.id].append((frag, ll))

        """
        Main loop
        """
        for multi_align in self._multi_align_instances:
            logger.debug(f"[{multi_align.canonical_marker.name}] Parsing alignment of reads "
                         f"({len(multi_align.forward_read_index_map)} forward, "
                         f"{len(multi_align.reverse_read_index_map)} reverse) "
                         f"into likelihoods.")
            # First, take care of the base markers (if applicable).
            for marker in multi_align.markers():
                if not self.model.bacteria_pop.contains_marker(marker):
                    continue

                for revcomp in [False, True]:
                    for read in multi_align.reads(revcomp=revcomp):
                        if not time_slice.contains_read(read.id):
                            continue

                        subseq, insertions, deletions, start_clip, end_clip = multi_align.get_aligned_reference_region(
                            marker,
                            read,
                            revcomp=revcomp
                        )

                        add_subseq_likelihood(subseq, read, insertions, deletions, revcomp, start_clip, end_clip)

            # Next, take care of the variant markers (if applicable).
            for variant in self.marker_variants_of(multi_align.canonical_marker):
                for read in time_slice:
                    for subseq, revcomp, insertions, deletions, start_clip, end_clip in variant.subseq_from_read(read):
                        add_subseq_likelihood(subseq, read, insertions, deletions, revcomp, start_clip, end_clip)

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
        def save_(path: Path, sparse_matrix: SparseMatrix):
            sparse_matrix.save(path)

        def load_(path: Path) -> SparseMatrix:
            return SparseMatrix.load(path, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)

        def callback_(matrix: SparseMatrix):
            _, counts_per_read = torch.unique(matrix.indices[1], sorted=False, return_inverse=False, return_counts=True)
            logger.debug(
                "Read-likelihood matrix (size {r} x {c}) has {nz} nonzero entries. "
                "(~{meanct:.2f} hits per read, density={dens:.1e}, physical({dev})={phys})".format(
                    r=matrix.size()[0],
                    c=matrix.size()[1],
                    nz=len(matrix.values),
                    meanct=counts_per_read.float().mean(),
                    dens=matrix.density(),
                    dev=str(matrix.values.device),
                    phys=convert_size(matrix.physical_size())
                )
            )

        jobs = [
            {
                "relative_filepath": "sparse_log_likelihoods_{}.npz".format(t_idx),
                "fn": lambda t: self.create_sparse_matrix(t),
                "call_args": [],
                "call_kwargs": {"t": t_idx},
                "save": save_,
                "load": load_,
                "success_callback": callback_
            }
            for t_idx in range(self.model.num_times())
        ]

        parallel = (self.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(self.num_cores))

            return Parallel(n_jobs=self.num_cores)(
                delayed(self.cache.call)(**cache_kwargs_t)
                for cache_kwargs_t in jobs
            )
        else:
            return [self.cache.call(**cache_kwargs_t) for cache_kwargs_t in jobs]

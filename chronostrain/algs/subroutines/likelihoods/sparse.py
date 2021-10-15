import torch
from typing import List, Dict, Set, Tuple, Iterator
from collections import defaultdict
import numpy as np

from joblib import Parallel, delayed

from chronostrain.algs.subroutines.alignment import CachedReadAlignments
from chronostrain.database import StrainDatabase
from chronostrain.model import Fragment, Marker
from chronostrain.util.sparse.sparse_tensor import SparseMatrix
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from .base import DataLikelihoods, AbstractLogLikelihoodComputer
from .likelihood_cache import LikelihoodMatrixCache

from chronostrain.config.logging import create_logger
from ... import MarkerVariant

logger = create_logger(__name__)


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
    def __init__(self, model: GenerativeModel, reads: TimeSeriesReads, db: StrainDatabase):
        super().__init__(model, reads)
        self._bwa_index_finished = False

        # Remember, these are alignments of reads to the database reference markers!
        self.cached_reference_alignments = CachedReadAlignments(self.reads, db)
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

    def marker_variants_of(self, marker: Marker) -> Iterator[MarkerVariant]:
        yield from self.variants_present[marker]

    def marker_isin_pop(self, marker: Marker) -> bool:
        return marker in self.markers_present

    def _compute_read_frag_alignments(self, t_idx: int) -> Dict[str, List[Tuple[Fragment, bool]]]:
        """
        Iterate through the timepoint's SamHandler instances (if the reads of t_idx came from more than one path).
        Each of these SamHandlers provides alignment information.
        :param t_idx: the timepoint index to use.
        :return: A defaultdict representing the map (Read ID) -> {Fragments that the read aligns to}
        """
        read_to_fragments: Dict[str, List[Tuple[Fragment, bool]]] = defaultdict(list)

        """
        Overall philosophy of this method: Keep things as simple as possible!
        Assume that indels have either been taken care of, either by passing in:
            1) appropriate MarkerVariant instances into the population, or
            2) include a complete reference cohort of Markers into the reference db 
            (but this is infeasible, as it requires knowing the ground truth on real data!)
        
        In particular, this means that we don't have to worry about indels.
        """
        for tgt_base_marker, alns in self.cached_reference_alignments.get_alignments(t_idx).items():
            for aln in alns:
                # First, add the likelihood for the fragment for the aligned base marker.
                if self.marker_isin_pop(tgt_base_marker):
                    """ We only care about one-to-one alignments (no insertions/deletions/clipping). """
                    marker_frag_seq = aln.marker_frag
                    try:
                        tgt_frag = self.model.get_fragment_space().get_fragment(marker_frag_seq)
                        read_to_fragments[aln.read_id].append((tgt_frag, aln.reverse_complemented))
                    except KeyError:
                        # Ignore these errors (see above note).
                        pass

                # Next, look up any variants of the base marker.
                for variant in self.marker_variants_of(tgt_base_marker):
                    variant_frag_seq = variant.subseq_from_base_marker_positions(
                        base_marker_start=aln.marker_start,
                        base_marker_end=aln.marker_end
                    )
                    variant_frag = self.model.get_fragment_space().get_fragment(variant_frag_seq)
                    logger.debug("Read: {}, Variant frag: {}".format(aln.read_seq, variant_frag))
                    try:
                        read_to_fragments[aln.read_id].append((variant_frag, aln.reverse_complemented))
                    except KeyError:
                        logger.debug("Line {} points to Read `{}`, but encountered KeyError. (Sam = {})".format(
                            aln.sam_line_no,
                            aln.read_id,
                            aln.sam_path,
                        ))
                        raise
        return read_to_fragments

    def create_sparse_matrix(self, t_idx) -> SparseMatrix:
        """
        For the specified time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param t_idx: The time point index to run this function on.
        :returns: A sparse representation of read indices, frag indices and the corresponding log-likelihoods,
        log P(read | frag).
        """
        # Perform alignment for approximate fine-grained search.
        read_to_fragments: Dict[str, List[Tuple[Fragment, bool]]] = self._compute_read_frag_alignments(t_idx)

        read_indices: List[int] = []
        frag_indices: List[int] = []
        log_likelihood_values: List[float] = []
        for read_idx, read in enumerate(self.reads[t_idx]):
            for frag, rev_comp in read_to_fragments[read.id]:
                read_indices.append(read_idx)
                frag_indices.append(frag.index)

                """
                0.5 chance of forward/reverse (rev_comp). This is an approximation of the dense version, 
                assuming that either p_forward or p_reverse is approximately zero.
                """
                log_likelihood_values.append(
                    self.model.error_model.compute_log_likelihood(frag, read, read_reverse_complemented=rev_comp)
                    - np.log(2)
                )

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
            dims=(self.model.get_fragment_space().size(), len(self.reads[t_idx]))
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

import torch
from typing import List, Dict, Set
from collections import defaultdict
import numpy as np

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed

from chronostrain.algs.subroutines.alignment import CachedReadAlignments
from chronostrain.algs.subroutines.read_cache import ReadsComputationCache
from chronostrain.database import StrainDatabase
from chronostrain.model import Fragment
from chronostrain.util.data_cache import ComputationCache, CacheTag
from chronostrain.util.sparse.sparse_tensor import SparseMatrix
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class DataLikelihoods(object):
    def __init__(
            self,
            model: GenerativeModel,
            data: TimeSeriesReads,
            read_likelihood_lower_bound: float = 1e-30
    ):
        """
        :param model:
        :param data:
        :param read_likelihood_lower_bound: Thresholds reads by this likelihood value.
            For each read, if the sum of all fragment-read likelihoods over all fragments does not exceed this value,
            the read is trimmed from the matrix (at the particular timepoint which it belongs to).
            (Note: passing '0' for this argument is the same as bypassing this filter.)
        """
        self.model = model
        self.data = data
        self.read_likelihood_lower_bound = read_likelihood_lower_bound

        log_likelihoods_tensors = self._likelihood_computer().compute_likelihood_tensors()
        self.matrices = [
            ll_tensor.exp() for ll_tensor in log_likelihoods_tensors
        ]
        self.retained_indices = self._trim()

    @abstractmethod
    def _likelihood_computer(self) -> 'AbstractLogLikelihoodComputer':
        raise NotImplementedError()

    def _trim(self) -> List[List[int]]:
        """
        Trims the likelihood matrices using the specified lower bound. Reads are removed if there are no fragments
        with likelihood greater than the lower bound. (This should ideally not happen if a stringent alignment-based
        filter was applied.)

        :return: List of the index of kept reads (exceeding the lower bound threshold).
        """
        read_indices = []
        for t_idx in range(self.model.num_times()):
            read_likelihoods_t = self.matrices[t_idx]
            sums = read_likelihoods_t.sum(dim=0)

            zero_indices = {i.item() for i in torch.where(sums <= self.read_likelihood_lower_bound)[0]}
            if len(zero_indices) > 0:
                logger.warn("[t = {}] Discarding reads with overall likelihood < {}: {}".format(
                    self.model.times[t_idx],
                    self.read_likelihood_lower_bound,
                    ",".join([str(read_idx) for read_idx in zero_indices])
                ))

                leftover_indices = [
                    read_idx
                    for read_idx in range(len(self.data[t_idx]))
                    if read_idx not in zero_indices
                ]
                read_indices.append(leftover_indices)

                if isinstance(read_likelihoods_t, SparseMatrix):
                    self.matrices[t_idx] = read_likelihoods_t.slice_columns(
                        leftover_indices
                    )
                else:
                    self.matrices[t_idx] = read_likelihoods_t[:, leftover_indices]
            else:
                read_indices.append(list(range(len(self.data[t_idx]))))
        return read_indices


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

    def _likelihood_computer(self) -> 'AbstractLogLikelihoodComputer':
        return SparseLogLikelihoodComputer(self.model, self.data, self.db)


class DenseDataLikelihoods(DataLikelihoods):
    def _likelihood_computer(self) -> 'AbstractLogLikelihoodComputer':
        return DenseLogLikelihoodComputer(self.model, self.data)


# ===================================================================
# ========================= Helper classes ==========================
# ===================================================================

class AbstractLogLikelihoodComputer(metaclass=ABCMeta):

    def __init__(self, model: GenerativeModel, reads: TimeSeriesReads):
        self.model = model
        self.reads = reads
        self.fragment_space = model.get_fragment_space()

    @abstractmethod
    def compute_likelihood_tensors(self) -> List[torch.Tensor]:
        """
        For each time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :returns: The array of likelihood tensors, indexed by timepoint indices.
        """
        pass


class DenseLogLikelihoodComputer(AbstractLogLikelihoodComputer):

    def __init__(self, model: GenerativeModel, reads: TimeSeriesReads):
        super().__init__(model, reads)

    def compute_matrix_single_timepoint(self, t_idx: int) -> List[List[float]]:
        """
        For the specified time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param t_idx: The time point index to run this function on.
        :returns: The array of likelihoods, stored as a length-F list of length-N_t lists.
        """
        ans = [
            [
                self.model.error_model.compute_log_likelihood(frag, read)
                for read in self.reads[t_idx]
            ]
            for frag in self.fragment_space.get_fragments()
        ]
        return ans

    def compute_likelihood_tensors(self) -> List[torch.Tensor]:
        # TODO: explicitly define save() and load() here to write directly to torch tensor files.
        #  (Right now, the behavior is to compute List[List[float]] and save/load from pickle.)

        logger.debug("Computing read-fragment likelihoods...")
        cache = ReadsComputationCache(self.reads)

        jobs = [
            {
                "filename": "log_likelihoods_{}.pkl".format(t_idx),
                "fn": lambda t: self.compute_matrix_single_timepoint(t),
                "args": [],
                "kwargs": {"t": t_idx}
            }
            for t_idx in range(self.model.num_times())
        ]

        parallel = (cfg.model_cfg.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))

            log_likelihoods_output = Parallel(n_jobs=cfg.model_cfg.num_cores)(
                delayed(cache.call)(cache_kwargs)
                for cache_kwargs in jobs
            )

            log_likelihoods_tensors = [
                torch.tensor(ll_array, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for ll_array in log_likelihoods_output
            ]
        else:
            log_likelihoods_tensors = [
                torch.tensor(cache.call(**cache_kwargs), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for cache_kwargs in jobs
            ]

        return log_likelihoods_tensors


class SparseLogLikelihoodComputer(AbstractLogLikelihoodComputer):
    def __init__(self, model: GenerativeModel, reads: TimeSeriesReads, db: StrainDatabase):
        super().__init__(model, reads)
        self._bwa_index_finished = False
        self.cached_alignments = CachedReadAlignments(self.reads, db)
        self.cache = ComputationCache(CacheTag(
            file_paths=[reads_t.src.paths for reads_t in reads],  # read files
            use_quality=cfg.model_cfg.use_quality_scores,
            markers=[
                marker
                for strain in model.bacteria_pop.strains
                for marker in strain.markers
            ]
        ))

    def _compute_read_frag_alignments(self, t_idx: int) -> Dict[str, Set[Fragment]]:
        """
        Iterate through the timepoint's SamHandler instances (if the reads of t_idx came from more than one path).
        Each of these SamHandlers provides alignment information.
        :param t_idx: the timepoint index to use.
        :return: A defaultdict representing the map (Read ID) -> {Fragments that the read aligns to}
        """
        read_to_fragments: Dict[str, Set[Fragment]] = defaultdict(set)
        for marker, alns in self.cached_alignments.get_alignments(t_idx).items():
            for aln in alns:
                # TODO - Future note: this might be a good starting place for handling/detecting indels.
                #  (Note: the below logic defaults to a KeyError.)
                #  (if one needs more control, handle this in the parse_alignments() function invoked in
                #  get_alignments().

                if aln.reverse_complemented:
                    logger.warning(
                        "Alignment ({f}) -- Found reverse-complemented alignment for read {r}.".format(
                            f=str(aln.sam_path),
                            r=aln.id
                        ))

                try:
                    aligned_frag = self.fragment_space.get_fragment(aln.marker_frag)
                except KeyError:
                    # This happens because the aligned frag is not a full alignment, either due to edge effects
                    # or indels. Our model does not handle these yet; see the note above.
                    continue

                try:
                    read_to_fragments[aln.read_id].add(aligned_frag)
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
        read_to_fragments: Dict[str, Set[Fragment]] = self._compute_read_frag_alignments(t_idx)

        read_indices: List[int] = []
        frag_indices: List[int] = []
        log_likelihood_values: List[float] = []
        for read_idx, read in enumerate(self.reads[t_idx]):
            for frag in read_to_fragments[read.id]:
                read_indices.append(read_idx)
                frag_indices.append(frag.index)
                log_likelihood_values.append(self.model.error_model.compute_log_likelihood(frag, read))

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
            dims=(self.fragment_space.size(), len(self.reads[t_idx]))
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

"""
 base.py
 Contains implementations of the proposed algorithms.
"""
import torch
from typing import List, Union
from pathlib import Path
from collections import defaultdict
import numpy as np

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed

from . import logger
from chronostrain.util.sparse.sparse_tensor import CoalescedSparseMatrix
from chronostrain.util.sam_handler import SamHandler
from chronostrain.util.external.bwa import bwa_index, bwa_mem
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel
from chronostrain.util.data_cache import CachedComputation, CacheTag


class DataLikelihoods(object):
    def __init__(
            self,
            model: GenerativeModel,
            data: TimeSeriesReads,
            use_sparse: bool,
            read_likelihood_lower_bound: float = 1e-30
    ):
        """
        :param model:
        :param data:
        :param use_sparse: Specifies whether to load sparse or dense versions of the matrices.
        :param read_likelihood_lower_bound: Thresholds reads by this likelihood value.
            For each read, if the sum of all fragment-read likelihoods over all fragments does not exceed this value,
            the read is trimmed from the matrix (at the particular timepoint which it belongs to).
            (Note: passing '0' for this argument is the same as bypassing this filter.)
        """
        self.model = model
        self.data = data
        self.use_sparse = use_sparse
        self.likelihoods_tensors: List[Union[torch.Tensor, CoalescedSparseMatrix]] = None
        self.retained_indices: List[List[int]] = None
        self.read_likelihood_lower_bound = read_likelihood_lower_bound

    @property
    def matrices(self) -> List[torch.Tensor]:
        if self.likelihoods_tensors is None:
            log_likelihoods_tensors = self._likelihood_computer().compute_likelihood_tensors()
            self.likelihoods_tensors = [
                ll_tensor.exp() for ll_tensor in log_likelihoods_tensors
            ]
            self.retained_indices = self._trim()
        return self.likelihoods_tensors

    def _likelihood_computer(self) -> 'AbstractLogLikelihoodComputer':
        if self.use_sparse:
            return SparseLogLikelihoodComputer(self.model, self.data)
        else:
            return DenseLogLikelihoodComputer(self.model, self.data)

    def _trim(self) -> List[List[int]]:
        """
        Trims the likelihood matrices using the specified lower bound.

        :return: List of the index of kept reads (exceeding the lower bound threshold).
        """
        read_indices = []
        for t_idx in range(self.model.num_times()):
            read_likelihoods_t = self.likelihoods_tensors[t_idx]
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

                if isinstance(read_likelihoods_t, CoalescedSparseMatrix):
                    self.likelihoods_tensors[t_idx] = read_likelihoods_t.slice_cols(
                        leftover_indices
                    )
                else:
                    self.likelihoods_tensors[t_idx] = read_likelihoods_t[:, leftover_indices]
            else:
                read_indices.append(list(range(len(self.data[t_idx]))))
        return read_indices


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self, model: GenerativeModel, data: TimeSeriesReads):
        self.model = model
        self.data = data
        self.data_likelihoods = DataLikelihoods(model, data, use_sparse=cfg.model_cfg.use_sparse)

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass


# ===================================================================
# ========================= Helper functions ========================
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
        logger.debug("Computing read-fragment likelihoods...")
        jobs: List[CachedComputation] = []

        cache_tag = CacheTag(
            file_paths=[reads_t.src for reads_t in self.reads],
            use_quality=cfg.model_cfg.use_quality_scores,
        )

        for t_idx in range(self.model.num_times()):
            jobs.append(
                CachedComputation(
                    self.compute_matrix_single_timepoint,
                    kwargs={"t_idx": t_idx},
                    filename="log_likelihoods_{}.pkl".format(t_idx),
                    cache_tag=cache_tag
                )
            )

        parallel = (cfg.model_cfg.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))

            with Parallel(n_jobs=cfg.model_cfg.num_cores) as parallel:
                log_likelihoods_output = [
                    parallel(delayed(job.call)())
                    for job in jobs
                ]
            log_likelihoods_tensors = [
                torch.tensor(ll_array, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for ll_array in log_likelihoods_output
            ]
        else:
            log_likelihoods_tensors = [
                torch.tensor(job.call(), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for job in jobs
            ]

        return log_likelihoods_tensors


class SparseLogLikelihoodComputer(AbstractLogLikelihoodComputer):
    def __init__(self, model, reads):
        super().__init__(model, reads)
        self.marker_reference_file = cfg.database_cfg.get_database().multifasta_file
        bwa_index(self.marker_reference_file)

    def _compute_read_frag_alignments(self, t_idx: int) -> defaultdict:
        read_to_fragments = defaultdict(set)
        alignment_output_path = Path(self.reads[0].src).parent / "all_alignments_{}.sam".format(t_idx)

        bwa_mem(
            output_path=alignment_output_path,
            reference_path=self.marker_reference_file,
            read_path=self.reads[t_idx].src,
            min_seed_length=20,
            report_all_alignments=True
        )

        sam_handler = SamHandler(alignment_output_path, self.marker_reference_file)
        for sam_line in sam_handler.mapped_lines():
            if sam_line.is_reverse_complemented:
                # TODO: Rest of the model does not handle reverse-complements, so we will skip those for now.
                continue

            # TODO - Future note: this might be a good starting place for handling/detecting indels.
            #  (since this part already handles the aligned_frag not being found.)
            #  (if one needs more control, handle this in the sam_line loop of _compute_read_frag_alignments().)
            try:
                aligned_frag = self.fragment_space.get_fragment(sam_line.fragment)
            except KeyError:
                # This happens because the aligned frag is not a full alignment, either due to edge effects or indels.
                # Our model does not handle these, but see the note above.
                continue

            read_id = sam_line.read
            try:
                read_to_fragments[read_id].add(aligned_frag)
            except KeyError:
                logger.debug("Problematic SAM line found: {}".format(
                    sam_line
                ))
                raise
        return read_to_fragments

    def create_sparse_matrix(self, t_idx) -> CoalescedSparseMatrix:
        """
        For the specified time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param t_idx: The time point index to run this function on.
        :returns: A sparse representation of read indices, frag indices and the corresponding log-likelihoods,
        log P(read | frag).
        """
        # Perform alignment for approximate fine-grained search.
        read_to_fragments = self._compute_read_frag_alignments(t_idx)

        read_indices: List[int] = []
        frag_indices: List[int] = []
        log_likelihood_values: List[float] = []
        for read_idx, read in enumerate(self.reads[t_idx]):
            for frag in read_to_fragments[read.nucleotide_content()]:
                read_indices.append(read_idx)
                frag_indices.append(frag.index)
                log_likelihood_values.append(self.model.error_model.compute_log_likelihood(frag, read))

        logger.debug(
            "Read-likelihood matrix (size {} x {}) has {} nonzero entries. (~{:.2f} hits per read)".format(
                self.fragment_space.size(),
                len(self.reads[t_idx]),
                len(log_likelihood_values),
                len(log_likelihood_values) / len(self.reads[t_idx])
            )
        )

        return CoalescedSparseMatrix(
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

    def compute_likelihood_tensors(self) -> List[CoalescedSparseMatrix]:
        logger.debug("Computing read-fragment likelihoods...")
        jobs: List[CachedComputation] = []

        cache_tag = CacheTag(
            file_paths=[reads_t.src for reads_t in self.reads],
            use_quality=cfg.model_cfg.use_quality_scores,
        )

        # Save each sparse tensor as a tuple of indices/values/shape into a compressed numpy file (.npz).
        def save_(path, sparse_matrix: CoalescedSparseMatrix):
            np.savez(
                path,
                sparse_indices=sparse_matrix.indices.cpu().numpy(),
                sparse_values=sparse_matrix.values.cpu().numpy(),
                matrix_shape=np.array([
                    sparse_matrix.rows,
                    sparse_matrix.columns
                ])
            )

        def load_(path) -> CoalescedSparseMatrix:
            data = np.load(path)
            size = data["matrix_shape"]
            return CoalescedSparseMatrix(
                indices=torch.tensor(
                    data['sparse_indices'],
                    device=cfg.torch_cfg.device,
                    dtype=torch.long
                ),
                values=torch.tensor(
                    data['sparse_values'],
                    device=cfg.torch_cfg.device,
                    dtype=torch.int
                ),
                dims=(size[0], size[1])
            )

        # Create an array of cached computation jobs.
        for t_idx in range(self.model.num_times()):
            jobs.append(
                CachedComputation(
                    self.create_sparse_matrix,
                    kwargs={"t_idx": t_idx},
                    filename="sparse_log_likelihoods_{}.npz".format(t_idx),
                    cache_tag=cache_tag,
                    save=save_,
                    load=load_
                )
            )

        parallel = (cfg.model_cfg.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))

            with Parallel(n_jobs=cfg.model_cfg.num_cores) as parallel:
                log_likelihoods_tensors = [
                    parallel(delayed(job.call)())
                    for job in jobs
                ]
            return log_likelihoods_tensors
        else:
            return [job_t.call() for job_t in jobs]

"""
 base.py
 Contains implementations of the proposed algorithms.
"""
import torch
from typing import List
from pathlib import Path
from collections import defaultdict
import numpy as np

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed

from . import logger
from chronostrain.util.math import mappings
from chronostrain.util.sparse.sparse_tensor import coalesced_sparse_tensor
from chronostrain.util.sam_handler import SamHandler, SamTags
from chronostrain.util.external.bwa import bwa_index, bwa_mem
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel
from chronostrain.util.data_cache import CachedComputation, CacheTag


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 cache_tag: CacheTag):
        self.model = model
        self.data = data
        self.cache_tag = cache_tag

        # Not sure which we will need. Use lazy initialization.
        self.read_likelihoods_loaded = False
        self.read_likelihoods_tensors: List[torch.Tensor] = []

        self.read_log_likelihoods_loaded = False
        self.read_log_likelihoods_tensors: List[torch.Tensor] = []

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @property
    def read_likelihoods(self) -> List[torch.Tensor]:
        if not self.read_log_likelihoods_loaded:
            self.read_log_likelihoods_tensors = self._compute_log_likelihoods()
            self.read_likelihoods_tensors = [
                mappings.exp(ll_tensor) for ll_tensor in self.read_log_likelihoods_tensors
            ]
            self.read_likelihoods_loaded = True
        return self.read_likelihoods_tensors

    @property
    def read_log_likelihoods(self) -> List[torch.Tensor]:
        if not self.read_log_likelihoods_loaded:
            self.read_log_likelihoods_tensors = self._compute_log_likelihoods()
            self.read_log_likelihoods_loaded = True
        return self.read_log_likelihoods_tensors

    def _compute_log_likelihoods(self):
        if cfg.model_cfg.use_sparse:
            likelihood_handler = SparseLogLikelhoodComputer(self.model, self.data)
        else:
            likelihood_handler = DenseLogLikelhoodComputer(self.model, self.data)
        return likelihood_handler.compute_likelihood_tensors()

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


class DenseLogLikelhoodComputer(AbstractLogLikelihoodComputer):

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

        for t_idx in range(self.model.num_times()):
            reads_t = self.reads[t_idx]
            cache_tag = CacheTag(
                file_path=reads_t.src,
                use_quality=cfg.model_cfg.use_quality_scores,
            )
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


class SparseLogLikelhoodComputer(AbstractLogLikelihoodComputer):
    def __init__(self, model, reads):
        super().__init__(model, reads)
        self.marker_reference_file = cfg.database_cfg.get_database().get_multifasta_file()
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
            read_id = sam_line[SamTags.Read]
            read_to_fragments[read_id].add(self.fragment_space.get_fragment(sam_line.get_fragment()))
        return read_to_fragments

    def sparse_matrix_specification(self, t_idx) -> torch.Tensor:
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
        return torch.sparse_coo_tensor(
            indices=torch.Tensor([frag_indices, read_indices]),
            values=log_likelihood_values,
            size=(self.fragment_space.size(), len(self.reads[t_idx])),
            device=cfg.torch_cfg.device,
            dtype=cfg.torch_cfg.default_dtype
        )

    def compute_likelihood_tensors(self) -> List[torch.Tensor]:
        logger.debug("Computing read-fragment likelihoods...")
        jobs: List[CachedComputation] = []

        # Save each sparse tensor as a tuple of indices/values/shape into a compressed numpy file (.npz).
        def save_(path, sparse_matrix: torch.Tensor):
            sparse_matrix = sparse_matrix.cpu()
            size = sparse_matrix.size()
            np.savez(
                path,
                sparse_indices=sparse_matrix.indices().numpy(),
                sparse_values=sparse_matrix.values().numpy(),
                matrix_shape=np.array([size[0], size[1]])
            )

        def load_(path) -> torch.Tensor:
            data = np.load(path)
            return torch.sparse_coo_tensor(
                indices=torch.Tensor(data['sparse_indices']),
                values=torch.Tensor(data['sparse_values']),
                size=torch.Size(data["matrix_shape"]),
                device=cfg.torch_cfg.device,
                dtype=cfg.torch_cfg.default_dtype
            )

        # Create an array of cached computation jobs.
        for t_idx in range(self.model.num_times()):
            reads_t = self.reads[t_idx]
            cache_tag = CacheTag(
                file_path=reads_t.src,
                use_quality=cfg.model_cfg.use_quality_scores,
            )
            jobs.append(
                CachedComputation(
                    self.sparse_matrix_specification,
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

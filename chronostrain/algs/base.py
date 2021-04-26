"""
 base.py
 Contains implementations of the proposed algorithms.
"""

import torch
from typing import List

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from tqdm import tqdm

from . import logger
import os
from chronostrain.util.sam_handler import SamHandler, SamTags
from chronostrain.util.external.bwa import bwa_index, bwa_mem
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel
from chronostrain.util.data_cache import CachedComputation, CacheTag
from chronostrain.util.benchmarking import current_time_millis, millis_elapsed


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
        self.marker_reference_file = cfg.database_cfg.get_database().get_multifasta_file()

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @property
    def read_likelihoods(self) -> List[torch.Tensor]:
        likelihood_handler = sparse_likelihood_handler(self.model, self.data, self.marker_reference_file)
        if not self.read_log_likelihoods_loaded:
            self.read_log_likelihoods_tensors = CachedComputation(likelihood_handler.compute_likelihood_tensors, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl"
            )
            
            self.read_likelihoods_tensors = [
                torch.exp(ll_tensor.to_dense()) for ll_tensor in self.read_log_likelihoods_tensors
            ]
            self.read_likelihoods_loaded = True
        return self.read_likelihoods_tensors

    @property
    def read_log_likelihoods(self) -> List[torch.Tensor]:
        likelihood_handler = sparse_likelihood_handler(self.model, self.data, self.marker_reference_file)
        if not self.read_log_likelihoods_loaded:
            self.read_log_likelihoods_tensors = CachedComputation(likelihood_handler.compute_likelihood_tensors, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl"
            )
            self.read_log_likelihoods_loaded = True
        return self.read_log_likelihoods_tensors


# ===================================================================
# ========================= Helper functions ========================
# ===================================================================

class log_likelihood_handler(metaclass=ABCMeta):

    @abstractmethod
    def create_matrix_spec(self, k):
        pass

    @abstractmethod
    def compute_likelihood_tensors(self, model: GenerativeModel, reads: TimeSeriesReads) -> List[torch.Tensor]:
        pass

class dense_likelihood_handler(log_likelihood_handler):
    def __init__(self, model, reads):
        self.model = model
        self.reads = reads
        self.fragment_space = model.get_fragment_space()

    def create_matrix_spec(self, k):
        """
        For the specified time point (t = t_k), evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param k: The time point index to run this function on.
        :returns: The array of likelihoods, stored as a length-F list of length-N_t lists.
        """
        start_t = current_time_millis()
        ans = [
            [
                self.model.error_model.compute_log_likelihood(frag, read)
                for read in self.reads[k]
            ] for frag in self.fragment_space.get_fragments()
        ]
        logger.debug("Chunk (k={k}) completed in {t:.1f} min.".format(
            k=k,
            t=millis_elapsed(start_t) / 60000
        ))
        return ans

    def compute_likelihood_tensors(self):
        start_time = current_time_millis()
        logger.debug("Computing read-fragment likelihoods...")

        parallel = (cfg.model_cfg.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))
            log_likelihoods_output = Parallel(n_jobs=cfg.model_cfg.num_cores)(delayed(self.create_matrix_spec)(k) for k in tqdm(range(len(self.model.times))))
            log_likelihoods_tensors = [
                torch.tensor(ll_array, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for ll_array in log_likelihoods_output
            ]
        else:
            log_likelihoods_tensors = [
                torch.tensor(self.create_matrix_spec(k), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for k in tqdm(range(len(self.model.times)))
            ]
        logger.debug("Computed fragment errors in {:1f} min.".format(millis_elapsed(start_time) / 60000))

        return log_likelihoods_tensors

class sparse_likelihood_handler(log_likelihood_handler):
    def __init__(self, model, reads, reference_file):
        self.model = model
        self.reads = reads
        self.fragment_space = model.get_fragment_space()
        print("LOADING DB for Ref File")
        self.read_reference_file = reference_file

    def compute_read_frag_alignments(self, k):
        self.read_frag_map = {}
        all_aligns_output_path = os.path.join(os.path.dirname(self.reads[0].src), "all_alignments_" + str(k) + ".sam")
        bwa_index(self.read_reference_file)
        bwa_mem(
            output_path=all_aligns_output_path,
            reference_path=self.read_reference_file,
            read_path=self.reads[k].src,
            min_seed_length=20,
            report_all_alignments=True
        )
        sam_handler = SamHandler(all_aligns_output_path, self.read_reference_file)
        for sam_line in sam_handler.mapped_lines():
            read = sam_line[SamTags.Read]
            aligned_frags = self.read_frag_map.get(read, [])
            aligned_frags.append(self.fragment_space.get_fragment(sam_line.get_fragment()))
            self.read_frag_map[read] = aligned_frags
        print(self.read_frag_map)

    def create_matrix_spec(self, k):
        start_t = current_time_millis()
        self.compute_read_frag_alignments(k)

        populated_indices = [[],[]]
        likelihoods = []
        for read_i in range(len(self.reads[k])):
            print("Attempting key: " + self.reads[k][read_i].nucleotide_content())
            for frag in self.read_frag_map.get(self.reads[k][read_i].nucleotide_content(), {}):
                populated_indices[1].append(read_i)
                populated_indices[0].append(frag.index)
                likelihoods.append(self.model.error_model.compute_log_likelihood(frag, self.reads[k][read_i]))
        logger.debug("Chunk (k={k}) completed in {t:.1f} min.".format(
            k=k,
            t=millis_elapsed(start_t) / 60000
        ))
        return (populated_indices, likelihoods, k)

    def compute_likelihood_tensors(self):
        start_time = current_time_millis()
        logger.debug("Computing read-fragment likelihoods...")

        log_likelihoods_output = []
        parallel = (cfg.model_cfg.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))
            log_likelihoods_output = Parallel(n_jobs=cfg.model_cfg.num_cores)(delayed(self.create_matrix_spec)(k) for k in tqdm(range(len(self.model.times))))
            
        else:
            log_likelihoods_output = [self.create_matrix_spec(k) for k in tqdm(range(len(self.model.times)))]
        log_likelihoods_tensors = [
            torch.sparse_coo_tensor(matrix_spec[0], matrix_spec[1], (self.fragment_space.size(), len(self.reads[matrix_spec[2]])), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
            for matrix_spec in log_likelihoods_output
        ]
        logger.debug("Computed fragment errors in {:1f} min.".format(millis_elapsed(start_time) / 60000))
        print("Log likelihoods tensors:")
        for tensor in log_likelihoods_tensors:
            print(tensor.size())
            print(tensor.to_dense())
            #print("Populated indices: " + str(len(tensor.values())))
        return log_likelihoods_tensors


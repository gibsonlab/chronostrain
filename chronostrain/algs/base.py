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
from chronostrain.util.sam_handler import SamHandler
from chronostrain.util.external import bwa_index, bwa_mem
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

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @property
    def read_likelihoods(self) -> List[torch.Tensor]:
        if not self.read_likelihoods_loaded:
            log_likelihoods = CachedComputation(compute_read_log_likelihoods, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl",
                model=self.model,
                reads=self.data
            )
            self.read_likelihoods_tensors = [
                torch.exp(ll_tensor) for ll_tensor in log_likelihoods
            ]
            self.read_likelihoods_loaded = True
        return self.read_likelihoods_tensors

    @property
    def read_log_likelihoods(self) -> List[torch.Tensor]:
        if not self.read_log_likelihoods_loaded:
            self.read_log_likelihoods_tensors = CachedComputation(compute_read_log_likelihoods, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl",
                model=self.model,
                reads=self.data
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
    def compile_likelihood_tensors(self, model: GenerativeModel, reads: TimeSeriesReads) -> List[torch.Tensor]:
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
    def __init__(self, model, reads):
        self.model = model
        self.reads = reads
        self.fragment_space = model.get_fragment_space
        self.read_reference_file = cfg.database_cfg.get_database().get_multifasta_file()

    def compute_read_frag_alignments(self, k):
        self.read_frag_map = {}
        bwa_index(self.read_reference_file)
        bwa_mem(
            output_path=,
            reference_path=self.read_reference_file,
            read_path=self.reads[k].src,
            min_seed_length=20,
            report_all_alignments=True
        )
        alignment_contents = self.read_alignment_file()
        for line in alignment_contents:
            read, fragment = parse_alignment(line, self.read_reference_file)
            frag_map = self.read_frag_map.get(read.Lower(), [])
            frag_map.append(fragment)
            self.read_frag_map[read.Lower()] = frag_map

    '''
    TODO: Move alignment utilities into centralized sam file handler for this and filter.py
    '''
    def read_alignment_file(self, alignment_path):
        aln_lines = []
        with open(alignment_path, 'r') as f:
            for line in f:
                if line[0] == '@':
                    continue
                aln_lines.append(line.strip().split('\t'))

    def parse_alignment(self, aln_line, reference_path):
        position_tag = int(aln_line[3])
        start_clip = self.find_start_clip(aln_line[5])
        reference_file_index = position_tag-start_clip-1
        # Fix fragment lookup for multifasta
        return aln_line[9], 

    def find_start_clip(self, cigar_tag):
        split_cigar = re.findall('\d+|\D+',cigar_tag)
        if split_cigar[1] == 'S':
            return int(split_cigar[0])
        return 0

    def create_matrix_spec(self, k):
        start_t = current_time_millis()
        self.compute_read_frag_alignments(k)

        populated_indices = [[],[]]
        likelihoods = []
        for read_i in range(len(reads[k])):
            for frag in self.read_frag_map.get(reads[k][i], {}):
                populated_indices[0].append(i)
                populated_indices[1].append(frag.index)
                likelihoods.append(self.model.error_model.compute_log_likelihood(frag, read))
        logger.debug("Chunk (k={k}) completed in {t:.1f} min.".format(
            k=k,
            t=millis_elapsed(start_t) / 60000
        ))
        return (populated_indices, likelihoods, k)

    def compute_likelihood_tensors(self):
        start_time = current_time_millis()
        logger.debug("Computing read-fragment likelihoods...")

        self.compute_frag_index_map()
        log_likelihoods_output = []
        parallel = (cfg.model_cfg.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))
            log_likelihoods_output = Parallel(n_jobs=cfg.model_cfg.num_cores)(delayed(self.create_matrix_spec)(k) for k in tqdm(range(len(self.model.times))))
            
        else:
            log_likelihoods_output = [self.create_matrix_spec(k) for k in tqdm(range(len(self.model.times)))]
        log_likelihoods_tensors = [
            torch.sparse_coo_tensor(matrix_spec[0], matrix_spec[1], (len(reads[matrix_spec[2]]), self.fragment_space.size()), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
            for matrix_spec in log_likelihoods_output
        ]
        logger.debug("Computed fragment errors in {:1f} min.".format(millis_elapsed(start_time) / 60000))
        return log_likelihoods_tensors


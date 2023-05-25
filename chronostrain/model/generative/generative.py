"""
 generative.py
 Contains classes for representing the generative model.
"""
from typing import List

import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse

from chronostrain.model.bacteria import Population
from chronostrain.model import FragmentSpace
from chronostrain.model.reads import AbstractErrorModel
from chronostrain.database import StrainDatabase
from chronostrain.config import cfg

from .fragment_frequencies import FragmentFrequencyComputer
from chronostrain.logging import create_logger

logger = create_logger(__name__)


class GenerativeModel:
    def __init__(self,
                 times: List[float],
                 tau_1_dof: float,
                 tau_1_scale: float,
                 tau_dof: float,
                 tau_scale: float,
                 bacteria_pop: Population,
                 fragments: FragmentSpace,
                 frag_negbin_n: float,
                 frag_negbin_p: float,
                 read_error_model: AbstractErrorModel,
                 min_overlap_ratio: float,
                 db: StrainDatabase,
                 ):
        """
        :param times: A list of time points.
        :param tau_1_dof: The scale-inverse-chi-squared DOF of the first time point.
        :param tau_1_scale: The scale-inverse-chi-squared scale of the first time point.
        :param tau_dof: The scale-inverse-chi-squared DOF of the rest of the Gaussian process.
        :param tau_scale: The scale-inverse-chi-squared scale of the rest of the Gaussian process.
        :param bacteria_pop: A Population instance consisting of the relevant strains.
        :param fragments: A FragmentSpace instance encapsulating all relevant fragments.
        :param read_error_model: An error model for the reads (an instance of AbstractErrorModel).
        """

        self.times: List[float] = times  # array of time points
        self.tau_1_dof: float = tau_1_dof
        self.tau_1_scale: float = tau_1_scale
        self.tau_dof: float = tau_dof
        self.tau_scale: float = tau_scale
        self.error_model: AbstractErrorModel = read_error_model
        self.bacteria_pop: Population = bacteria_pop
        self.fragments: FragmentSpace = fragments
        self.frag_nbinom_n = frag_negbin_n
        self.frag_nbinom_p = frag_negbin_p

        self.min_overlap_ratio: float = min_overlap_ratio

        self.db = db
        self._frag_freqs_sparse = None
        self._frag_freqs_dense = None

        logger.debug(f"Model has inverse temperature = {cfg.model_cfg.inverse_temperature}")
        self.latent_conversion = lambda x: jax.nn.softmax(cfg.model_cfg.inverse_temperature * x, axis=-1)

        # self.log_latent_conversion = lambda x: torch.log(sparsemax(x, dim=-1))
        self.log_latent_conversion = lambda x: jax.nn.log_softmax(cfg.model_cfg.inverse_temperature * x, axis=-1)

        self.dt_sqrt_inverse = np.power(np.array(
            [
                self.dt(t_idx)
                for t_idx in range(1, self.num_times())
            ],
            dtype=cfg.engine_cfg.dtype
        ), -0.5)

    def num_times(self) -> int:
        return len(self.times)

    def num_strains(self) -> int:
        return self.bacteria_pop.num_strains()

    def num_fragments(self) -> int:
        return len(self.fragments)

    @property
    def fragment_frequencies_sparse(self) -> jsparse.BCOO:
        """
        Outputs the (F x S) matrix representing the strain-specific fragment (LOG-)frequencies.
        Is a wrapper for Population.construct_strain_fragment_frequencies().
        (Corresponds to the matrix "W" in writeup.)
        """
        if self._frag_freqs_sparse is None:
            self._frag_freqs_sparse = FragmentFrequencyComputer(
                frag_nbinom_n=self.frag_nbinom_n,
                frag_nbinom_p=self.frag_nbinom_p,
                db=self.db,
                fragments=self.fragments,
                min_overlap_ratio=self.min_overlap_ratio,
                dtype=cfg.engine_cfg.dtype
            ).get_frequencies(self.fragments, self.bacteria_pop)
        return self._frag_freqs_sparse

    @property
    def fragment_frequencies_dense(self) -> np.ndarray:
        """
        Outputs the (F x S) matrix representing the strain-specific fragment (LOG-)frequencies.
        Is a wrapper for Population.construct_strain_fragment_frequencies().
        (Corresponds to the matrix "W" in writeup.)
        """
        raise NotImplementedError("TODO implement `DenseFragmentFrequencyComputer` class.")

    def log_likelihood_x(self, x: np.ndarray) -> np.ndarray:
        """
        Given an (T x N x S) tensor where N = # of instances/samples of X, compute the N different log-likelihoods.
        """
        if len(x.shape) == 2:
            r, c = x.shape
            x = x.reshape(r, 1, c)
        return self.log_likelihood_x_jeffreys_prior(x)

    def log_likelihood_x_jeffreys_prior(self, x: np.ndarray) -> np.ndarray:
        """
        Implementation of log_likelihood_x using Jeffrey's prior (for the Gaussian with known mean) for the variance.
        Assumes that the shape of X is constant (and only returns the non-constant part.)
        """
        n_times, _, n_strains = x.shape

        ll_first = -0.5 * n_strains * np.log(np.square(
            x[0, :, :]
        ).sum(axis=-1))
        ll_rest = -0.5 * (n_times - 1) * n_strains * np.log(np.square(
            np.expand_dims(self.dt_sqrt_inverse, axis=[1, 2]) * np.diff(x, n=1, axis=0)
        ).sum(axis=0).sum(axis=-1))
        return ll_first + ll_rest

    def dt(self, time_idx: int) -> float:
        """
        Return the k-th time increment, t_k - t_{k-1}.
        Raises an error if k == 0.
        :param time_idx: The index (k).
        :return:
        """
        if time_idx == 0 or time_idx >= self.num_times():
            raise IndexError("Can't get time increment at index {}.".format(time_idx))
        else:
            return self.times[time_idx] - self.times[time_idx - 1]

    def time_scaled_variance(self, time_idx: int, var_1: float, var: float) -> float:
        """
        Return the k-th time incremental variance.
        :param time_idx: the index to query (corresponding to k).
        :param var_1: The value of (tau_1)^2, the variance-scaling term of the first observed timepoint.
        :param var: The value of (tau)^2, the variance-scaling term of the underlying Gaussian process.
        :return: the kth variance term (t_k - t_(k-1)) * tau^2.
        """

        if time_idx == 0:
            return var_1
        elif time_idx < len(self.times):
            return var * self.dt(time_idx)
        else:
            raise IndexError("Can't reference time at index {}.".format(time_idx))

    # def strain_abundance_to_frag_abundance(self, strain_abundances: np.ndarray) -> np.ndarray:
    #     """
    #     Convert strain abundance to fragment abundance, via the matrix multiplication F = WZ.
    #     Assumes strain_abundance is an S x T tensor, so that the output is an F x T tensor.
    #     """
    #     if cfg.model_cfg.use_sparse:
    #         return self.fragment_frequencies_sparse.exp().dense_mul(strain_abundances)
    #     else:
    #         return self.fragment_frequencies_dense.exp().mm(strain_abundances)

    # def sample_reads(
    #         self,
    #         frag_abundances: torch.Tensor,
    #         num_samples: int = 1,
    #         read_length: int = 150,
    #         metadata: str = "") -> List[SequenceRead]:
    #     """
    #     Given a set of fragments and their time indexed frequencies (based on the current time
    #     index strain abundances and the fragments' relative frequencies in each strain's sequence.),
    #     generate a set of noisy fragment reads where the read fragments are selected in proportion
    #     to their time indexed frequencies and the outputted base pair at location i of each selected
    #     fragment is chosen from a probability distribution condition on the actual base pair at location
    #     i and the quality score at location i in the generated quality score vector for the
    #     selected fragment.
    #
    #     :param - frag_abundances: a tensor of floats representing a probability distribution over the fragments
    #     :param - num_samples: the number of samples to be taken.
    #     :return: a list of strings representing a noisy reads of the set of input fragments
    #     """
    #
    #     frag_indexed_samples = torch.multinomial(
    #         frag_abundances,
    #         num_samples=num_samples,
    #         replacement=True
    #     )
    #
    #     frag_samples = []
    #
    #     from .. import construct_fragment_space_uniform_length
    #     fragments = construct_fragment_space_uniform_length(read_length, self.bacteria_pop)
    #
    #     # Draw a read from each fragment.
    #     for i in range(num_samples):
    #         frag = fragments.get_fragment_by_index(frag_indexed_samples[i].item())
    #         frag_samples.append(self.error_model.sample_noisy_read(
    #             read_id="SimRead_{}".format(i),
    #             fragment=frag,
    #             metadata="{}|{}".format(
    #                 metadata, "|".join(frag.metadata)
    #             )
    #         ))
    #
    #     return frag_samples

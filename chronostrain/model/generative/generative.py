"""
 generative.py
 Contains classes for representing the generative model.
"""
from typing import List, Tuple, Callable

from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import geom

from chronostrain.model.bacteria import Population
from chronostrain.model.fragments import FragmentSpace
from chronostrain.model.reads import AbstractErrorModel, SequenceRead
from chronostrain.model.io import TimeSeriesReads, TimeSliceReads
from chronostrain.util.math.distributions import *
from chronostrain.util.sparse import RowSectionedSparseMatrix
from .fragment_frequencies import SparseFragmentFrequencyComputer

from chronostrain.config.logging import create_logger
from chronostrain.database import StrainDatabase

logger = create_logger(__name__)


# noinspection PyPep8Naming,DuplicatedCode
class GenerativeModel:
    def __init__(self,
                 times: List[float],
                 mu: torch.Tensor,
                 tau_1_dof: float,
                 tau_1_scale: float,
                 tau_dof: float,
                 tau_scale: float,
                 bacteria_pop: Population,
                 fragments: FragmentSpace,
                 frag_adapter_p: float,
                 read_error_model: AbstractErrorModel,
                 max_read_len: int,
                 min_overlap_ratio: float,
                 db: StrainDatabase):
        """
        :param times: A list of time points.
        :param mu: The prior mean E[X_1] of the first time point.
        :param tau_1_dof: The scale-inverse-chi-squared DOF of the first time point.
        :param tau_1_scale: The scale-inverse-chi-squared scale of the first time point.
        :param tau_dof: The scale-inverse-chi-squared DOF of the rest of the Gaussian process.
        :param tau_scale: The scale-inverse-chi-squared scale of the rest of the Gaussian process.
        :param bacteria_pop: A Population instance consisting of the relevant strains.
        :param fragments: A FragmentSpace instance encapsulating all relevant fragments.
        :param read_error_model: An error model for the reads (an instance of AbstractErrorModel).
        """

        self.times: List[float] = times  # array of time points
        self.mu: torch.Tensor = mu  # mean for X_1
        self.tau_1_dof: float = tau_1_dof
        self.tau_1_scale: float = tau_1_scale
        self.tau_dof: float = tau_dof
        self.tau_scale: float = tau_scale
        self.error_model: AbstractErrorModel = read_error_model
        self.bacteria_pop: Population = bacteria_pop
        self.fragments: FragmentSpace = fragments
        self.frag_length_logpmf: Callable[[int], float] = lambda k: geom.logpmf(
            k=max_read_len - k + 1,
            p=frag_adapter_p
        )

        self.max_read_len: int = max_read_len
        self.min_overlap_ratio: float = min_overlap_ratio

        self.db = db
        self._frag_freqs_sparse = None
        self._frag_freqs_dense = None

    def num_times(self) -> int:
        return len(self.times)

    def num_strains(self) -> int:
        return self.bacteria_pop.num_strains()

    def num_fragments(self) -> int:
        return self.fragments.size()

    @property
    def fragment_frequencies_sparse(self) -> RowSectionedSparseMatrix:
        """
        Outputs the (F x S) matrix representing the strain-specific fragment (LOG-)frequencies.
        Is a wrapper for Population.construct_strain_fragment_frequencies().
        (Corresponds to the matrix "W" in writeup.)
        """
        if self._frag_freqs_sparse is None:
            self._frag_freqs_sparse = SparseFragmentFrequencyComputer(
                length_logpmf=self.frag_length_logpmf,
                db=self.db,
                min_overlap_ratio=self.min_overlap_ratio,
                max_read_len=self.max_read_len
            ).get_frequencies(self.fragments, self.bacteria_pop)
        return self._frag_freqs_sparse

    @property
    def fragment_frequencies_dense(self) -> torch.Tensor:
        """
        Outputs the (F x S) matrix representing the strain-specific fragment (LOG-)frequencies.
        Is a wrapper for Population.construct_strain_fragment_frequencies().
        (Corresponds to the matrix "W" in writeup.)
        """
        raise NotImplementedError("TODO implement `DenseFragmentFrequencyComputer` class.")

    def log_likelihood_x(self, X: torch.Tensor) -> torch.Tensor:
        """
        Given an (T x N x S) tensor where N = # of instances/samples of X, compute the N different log-likelihoods.
        """
        if len(X.size()) == 2:
            r, c = X.size()
            X = X.view(r, 1, c)
        return self.log_likelihood_x_sics_prior(X)

    def log_likelihood_x_halfcauchy_prior(self, X: torch.Tensor) -> torch.Tensor:
        """
        Implementation of log_likelihood_x using HalfCauchy prior for the variance.
        """
        log_likelihood_first = HalfCauchyVarianceGaussian(
            mean=self.mu,
            cauchy_scale=100.0,
            n_samples=200
        ).empirical_log_likelihood(x=X[0, :, :])

        collapsed_size = (self.num_times() - 1) * self.num_strains()
        n_samples = X.size()[1]

        dt_sqrt_inverse = torch.tensor(
            [
                self.dt(t_idx)
                for t_idx in range(1, self.num_times())
            ],
            device=cfg.torch_cfg.device
        ).pow(-0.5)
        diffs = (X[1:, :, ] - X[:-1, :, ]) * dt_sqrt_inverse.unsqueeze(1).unsqueeze(2)

        log_likelihood_rest = HalfCauchyVarianceGaussian(
            mean=torch.zeros(n_samples, collapsed_size, dtype=cfg.torch_cfg.default_dtype),
            cauchy_scale=1.0,
            n_samples=200
        ).empirical_log_likelihood(
            x=diffs.transpose(0, 1).reshape(n_samples, collapsed_size)
        )

        return log_likelihood_first + log_likelihood_rest

    def log_likelihood_x_uniform_prior(self, X: torch.Tensor) -> torch.Tensor:
        """
        Implementation of log_likelihood_x using Uniform prior for the variance.
        """
        log_likelihood_first = UniformVarianceGaussian(
            mean=self.mu,
            lower=0.1,
            upper=20.0,
            steps=25
        ).log_likelihood(x=X[0, :, :])

        collapsed_size = (self.num_times() - 1) * self.num_strains()
        n_samples = X.size()[1]

        dt_sqrt_inverse = torch.tensor(
            [
                self.dt(t_idx)
                for t_idx in range(1, self.num_times())
            ]
        ).pow(-0.5)
        diffs = (X[1:, :, ] - X[:-1, :, ]) * dt_sqrt_inverse.unsqueeze(1).unsqueeze(2)

        log_likelihood_rest = UniformVarianceGaussian(
            mean=torch.zeros(n_samples, collapsed_size, dtype=cfg.torch_cfg.default_dtype),
            lower=0.1,
            upper=20.0,
            steps=25
        ).log_likelihood(
            x=diffs.transpose(0, 1).reshape(n_samples, collapsed_size)
        )

        return log_likelihood_first + log_likelihood_rest

    def log_likelihood_x_jeffreys_prior(self, X: torch.Tensor) -> torch.Tensor:
        """
        Implementation of log_likelihood_x using Jeffrey's prior (for the Gaussian with known mean) for the variance.
        """
        log_likelihood_first = JeffreysGaussian(mean=self.mu).log_likelihood(x=X[0, :, :])

        collapsed_size = (self.num_times() - 1) * self.num_strains()
        n_samples = X.size()[1]

        dt_sqrt_inverse = torch.tensor(
            [
                self.dt(t_idx)
                for t_idx in range(1, self.num_times())
            ]
        ).pow(-0.5)
        diffs = (X[1:, :, ] - X[:-1, :, ]) * dt_sqrt_inverse.unsqueeze(1).unsqueeze(2)

        log_likelihood_rest = JeffreysGaussian(
            mean=torch.zeros(n_samples, collapsed_size, dtype=cfg.torch_cfg.default_dtype)
        ).log_likelihood(
            x=diffs.transpose(0, 1).reshape(n_samples, collapsed_size)
        )

        return log_likelihood_first + log_likelihood_rest

    def log_likelihood_x_sics_prior(self, X: torch.Tensor) -> torch.Tensor:
        ans = torch.zeros(size=[X[0].size()[0]], device=X[0].device)
        X_prev = None
        for t_idx, X_t in enumerate(X):
            ans = ans + self.log_likelihood_xt_sics_prior_helper(
                t_idx=t_idx,
                X=X_t,
                X_prev=X_prev
            )
            X_prev = X_t
        return ans

    def log_likelihood_xt_sics_prior_helper(self,
                                            t_idx: int,
                                            X: torch.Tensor,
                                            X_prev: torch.Tensor):
        """
        Computes the Gaussian + Data likelihood at timepoint t, given previous timepoint X_prev.

        :param t_idx: the time index (0 thru T-1)
        :param X: (N x S) tensor of time (t) samples.
        :param X_prev: (N x S) tensor of time (t-1) samples. (None if t=0).
        :return: The joint log-likelihood p(X_t, Reads_t | X_{t-1}).
        """
        # Gaussian part
        N = X.size()[0]
        if t_idx == 0:
            center = self.mu.repeat(N, 1)
            dof = self.tau_1_dof
            scale = self.tau_1_scale
            dt = 1
        else:
            center = X_prev
            dof = self.tau_dof
            scale = self.tau_scale
            dt = self.dt(t_idx)

        return SICSGaussian(mean=center, dof=dof, scale=scale).log_likelihood(x=X, t=dt)

    def sample_abundances_and_reads(
            self,
            read_depths: List[int]
    ) -> Tuple[torch.Tensor, TimeSeriesReads]:
        """
        Generate a time-indexed list of read collections and strain abundances.

         :param read_depths: the number of reads per time point. A time-indexed array of ints.
         :return: a tuple of (abundances, reads) where
             strain_abundances is
                 a time-indexed list of 1D numpy arrays of fragment abundances based on the
                 corresponding time index strain abundances and the fragments' relative
                 frequencies in each strain's sequence.
             reads is
                a time-indexed list of lists of SequenceRead objects, where the i-th
                inner list corresponds to a reads taken at time index i (self.times[i])
                and contains num_samples[i] read objects.
        """
        if len(read_depths) != len(self.times):
            raise ValueError("Length of num_samples ({}) must agree with number of time points ({})".format(
                len(read_depths), len(self.times))
            )

        abundances = self.sample_abundances()
        reads = self.sample_timed_reads(abundances, read_depths)
        return abundances, reads

    def sample_timed_reads(self,
                           abundances: torch.Tensor,
                           read_depths: List[int],
                           read_length: int = 150
                           ) -> TimeSeriesReads:
        S = self.num_strains()
        F = self.num_fragments()
        num_timepoints = len(read_depths)

        logger.debug("Sampling reads, conditioned on abundances.")

        if abundances.size()[0] != len(self.times):
            raise ValueError(
                "Argument abundances (len {}) must must have specified number of time points (len {})".format(
                    abundances.size()[0], len(self.times)
                )
            )

        # For each time point, convert to fragment abundances and sample each read.
        time_slices = []

        for t in range(num_timepoints):
            read_depth = read_depths[t]
            strain_abundance = abundances[t]
            frag_abundance = self.strain_abundance_to_frag_abundance(strain_abundance.view(S, 1)).view(F)
            reads_arr = self.sample_reads(frag_abundance, read_depth,
                                          read_length=read_length,
                                          metadata="SIM_t{}".format(self.times[t]))
            time_slices.append(TimeSliceReads(
                reads=reads_arr,
                time_point=self.times[t],
                read_depth=read_depth
            ))

        return TimeSeriesReads(time_slices)

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

    def _sample_brownian_motion(self) -> torch.Tensor:
        """
        Generates an S-dimensional brownian motion, S = # of strains, centered at mu.
        Initial covariance is tau_1, while subsequent variance scaling is tau.
        """
        brownian_motion = []
        covariance = torch.eye(list(self.mu.size())[0], device=cfg.torch_cfg.device)  # Initialize covariance matrix
        center = self.mu  # Initialize mean vector.
        var_1 = ScaleInverseChiSquared(dof=self.tau_1_dof, scale=self.tau_1_scale).sample().item()
        var = ScaleInverseChiSquared(dof=self.tau_dof, scale=self.tau_scale).sample().item()

        for time_idx in range(len(self.times)):
            variance_scaling = self.time_scaled_variance(time_idx, var_1=var_1, var=var)
            center = MultivariateNormal(loc=center, covariance_matrix=variance_scaling * covariance).sample()
            brownian_motion.append(center)

        return torch.stack(brownian_motion, dim=0)

    def sample_abundances(self) -> torch.Tensor:
        """
        Samples abundances from a Gaussian Process.

        :return: A T x S tensor; each row is an abundance profile for a time point.
        """
        gaussians = self._sample_brownian_motion()
        return torch.softmax(gaussians, dim=1)

    def strain_abundance_to_frag_abundance(self, strain_abundances: torch.Tensor) -> torch.Tensor:
        """
        Convert strain abundance to fragment abundance, via the matrix multiplication F = WZ.
        Assumes strain_abundance is an S x T tensor, so that the output is an F x T tensor.
        """
        if cfg.model_cfg.use_sparse:
            return self.fragment_frequencies_sparse.exp().dense_mul(strain_abundances)
        else:
            return self.fragment_frequencies_dense.exp().mm(strain_abundances)

    def sample_reads(
            self,
            frag_abundances: torch.Tensor,
            num_samples: int = 1,
            read_length: int = 150,
            metadata: str = "") -> List[SequenceRead]:
        """
        Given a set of fragments and their time indexed frequencies (based on the current time
        index strain abundances and the fragments' relative frequencies in each strain's sequence.),
        generate a set of noisy fragment reads where the read fragments are selected in proportion
        to their time indexed frequencies and the outputted base pair at location i of each selected
        fragment is chosen from a probability distribution condition on the actual base pair at location
        i and the quality score at location i in the generated quality score vector for the
        selected fragment.

        :param - frag_abundances: a tensor of floats representing a probability distribution over the fragments
        :param - num_samples: the number of samples to be taken.
        :return: a list of strings representing a noisy reads of the set of input fragments
        """

        frag_indexed_samples = torch.multinomial(
            frag_abundances,
            num_samples=num_samples,
            replacement=True
        )

        frag_samples = []
        from .util import construct_fragment_space_uniform_length
        fragments = construct_fragment_space_uniform_length(read_length, self.bacteria_pop)

        # Draw a read from each fragment.
        for i in range(num_samples):
            frag = fragments.get_fragment_by_index(frag_indexed_samples[i].item())
            frag_samples.append(self.error_model.sample_noisy_read(
                read_id="SimRead_{}".format(i),
                fragment=frag,
                metadata="{}|{}".format(
                    metadata, "|".join(frag.metadata)
                )
            ))

        return frag_samples

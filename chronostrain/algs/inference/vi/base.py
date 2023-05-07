from abc import abstractmethod, ABC, ABCMeta
from pathlib import Path
from typing import *

import jax.numpy as np

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.util.math import log_spspmm_exp
from chronostrain.util.optimization import LossOptimizer

from .. import AbstractModelSolver
from .util import divide_columns_into_batches_sparse

from chronostrain.logging import create_logger
logger = create_logger(__name__)


_GENERIC_PARAM_TYPE = Dict[str, np.ndarray]
_GENERIC_GRAD_TYPE = _GENERIC_PARAM_TYPE  # the two types usually tend to match.
_GENERIC_SAMPLE_TYPE = Dict[Any, np.ndarray]


class AbstractPosterior(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed, simplex-valued (T x N x S) abundance tensor.
        """
        pass

    @abstractmethod
    def mean(self) -> np.ndarray:
        """
        Returns the mean of this posterior distribution.
        :return: A time-indexed (T x S) abundance tensor.
        """
        pass

    @abstractmethod
    def log_likelihood(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def save(self, target_path: Path):
        pass


class AbstractReparametrizedPosterior(AbstractPosterior, ABC):
    def log_likelihood(self, samples: np.ndarray, params: _GENERIC_PARAM_TYPE = None) -> np.ndarray:
        pass

    def sample(self, num_samples: int = 1) -> np.ndarray:
        return self.reparametrize(self.random_sample(num_samples=num_samples))

    @abstractmethod
    def get_parameters(self) -> _GENERIC_PARAM_TYPE:
        raise NotImplementedError()

    @abstractmethod
    def mean(self, params: _GENERIC_PARAM_TYPE = None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def entropy(self, params: _GENERIC_PARAM_TYPE = None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def random_sample(self, num_samples: int) -> _GENERIC_SAMPLE_TYPE:
        """
        Return randomized samples (before reparametrization.)
        :param num_samples:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def set_parameters(self, params: _GENERIC_PARAM_TYPE):
        """
        Store the value of these params internally as the state of this posterior.
        :param params: A list of parameter arrays (the implementation should decide the ordering.)
        :return:
        """
        pass

    def reparametrize(self, random_samples: _GENERIC_SAMPLE_TYPE, params: _GENERIC_PARAM_TYPE = None) -> np.ndarray:
        raise NotImplementedError()


class AbstractADVI(ABC):
    """
    An abstraction of the autograd-driven (black-box) VI implementation.
    """

    def __init__(
            self,
            posterior: AbstractReparametrizedPosterior,
            optimizer: LossOptimizer
    ):
        self.posterior = posterior
        self.optim = optimizer
        self.optim.initialize(self.posterior.get_parameters())

    def optimize(self,
                 num_epochs: int = 1,
                 iters: int = 50,
                 num_samples: int = 150,
                 min_lr: float = 1e-4,
                 callbacks: Optional[List[Callable[[int, np.ndarray, float], None]]] = None):
        time_est = RuntimeEstimator(total_iters=num_epochs, horizon=10)
        elbo_value = 0.0

        logger.info("Starting ELBO optimization.")

        for epoch in range(1, num_epochs + 1, 1):
            # =========== Necessary preprocessing for new epoch.
            self.advance_epoch()

            # =========== Store ELBO values for reporting.
            epoch_elbos = []
            time_est.stopwatch_click()
            for it in range(1, iters + 1, 1):
                # ========== Perform optimization for each iteration.
                # random nodes pre-reparam
                random_samples = self.posterior.random_sample(num_samples=num_samples)

                # optimize and output ELBO.
                elbo_value = self.optimize_step(random_samples)

                # Store for reporting.
                epoch_elbos.append(elbo_value)

            # ===========  End of epoch
            epoch_elbo_avg = np.mean(epoch_elbos).item()

            if callbacks is not None:
                random_samples = self.posterior.random_sample(num_samples=num_samples)
                reparam_samples = self.posterior.reparametrize(random_samples)
                for callback in callbacks:
                    callback(epoch, reparam_samples, elbo_value)

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            logger.info(
                "Epoch {epoch} | time left: {t:.2f} min. | Average ELBO = {elbo:.2f} | LR = {lr}".format(
                    epoch=epoch,
                    t=time_est.time_left() / 60000,
                    elbo=epoch_elbo_avg,
                    lr=self.optim.current_learning_rate()
                )
            )

            if self.optim.current_learning_rate() < min_lr:
                logger.info("Stopping criteria met after {} epochs.".format(epoch))
                break

        # ========== End of optimization
        logger.info("Finished.")

    @abstractmethod
    def advance_epoch(self):
        """
        Do any pre-processing required for a new epoch (e.g. mini-batch data).
        Called at the start of every epoch, including the first one.
        """
        raise NotImplementedError()

    @abstractmethod
    def elbo_with_grad(
            self,
            params: _GENERIC_PARAM_TYPE,
            random_samples: _GENERIC_SAMPLE_TYPE
    ) -> Tuple[np.ndarray, _GENERIC_GRAD_TYPE]:
        """
        :return: The ELBO value, logically separated into `batches` if necessary.
        In implementations, save memory by yielding batches instead of returning a list.
        """
        raise NotImplementedError()

    def optimize_step(
            self,
            random_samples: _GENERIC_SAMPLE_TYPE
    ) -> np.ndarray:
        elbo_value, elbo_grad = self.elbo_with_grad(self.optim.params, random_samples)
        assert self.optim.grad_sign == -1
        self.optim.update(elbo_value, elbo_grad)
        return np.concatenate(elbo_value)


class AbstractADVISolver(AbstractModelSolver, AbstractADVI, ABC):
    """
    A basic implementation of ADVI estimating the posterior p(X|R), with fragments
    F marginalized out.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 posterior: AbstractReparametrizedPosterior,
                 optimizer: LossOptimizer,
                 read_batch_size: int = 5000):
        AbstractModelSolver.__init__(
            self,
            model,
            data,
            db
        )
        self.read_batch_size = read_batch_size

        AbstractADVI.__init__(self, posterior, optimizer)

        logger.debug("Initializing ADVI data structures.")
        if not cfg.model_cfg.use_sparse:
            raise NotImplementedError("ADVI only supports sparse data structures.")

        # (S x R) matrices: Contains P(R = r | S = s) for each read r, strain s.
        self.batches: List[List[np.ndarray]] = [
            [] for _ in range(model.num_times())
        ]
        self.total_reads: int = 0

        # Precompute likelihood products.
        logger.debug("Precomputing likelihood marginalization.")
        data_likelihoods = self.data_likelihoods
        for t_idx in range(model.num_times()):
            data_ll_t = data_likelihoods.reduce_supported_fragments(
                data_likelihoods.matrices[t_idx],
                t_idx,
                fragment_dim=0
            )
            frag_freqs_reduced = data_likelihoods.reduce_supported_fragments(
                self.model.fragment_frequencies_sparse,
                t_idx,
                fragment_dim=0
            )

            for batch_idx, data_t_batch in enumerate(divide_columns_into_batches_sparse(data_ll_t, self.read_batch_size)):
                # ========= Pre-compute likelihood calculations.
                strain_batch_lls_t = log_spspmm_exp(
                    frag_freqs_reduced.T,  # (S x F_), note the transpose!
                    data_ll_t  # F_ x R_batch
                )  # (S x R_batch)

                # ========= Locate and filter out reads with no good alignments.
                bad_indices = {
                    int(x)
                    for x in np.where(
                        np.equal(
                            np.sum(~np.isinf(strain_batch_lls_t), axis=0),
                            0
                        )
                    )[0]
                }
                if len(bad_indices) > 0:
                    logger.warning("(t = {}, batch {}) Found {} reads without good alignments.".format(
                        t_idx, batch_idx, len(bad_indices)
                    ))

                # =========== Locate and filter out reads that are non-discriminatory
                # (e.g. are not helpful for inference/contribute little to the posterior)
                # nondisc_indices = {
                #     int(x)
                #     for x in torch.where(
                #         torch.le(
                #             torch.var(strain_batch_lls_t, dim=0, keepdim=False).cpu(),
                #             torch.tensor(0.01)
                #         )
                #     )[0]
                # }
                # if len(nondisc_indices) > 0:
                #     logger.warning("(t = {}, batch {}) Found {} non-discriminatory reads.".format(
                #         t_idx, batch_idx, len(nondisc_indices)
                #     ))
                #
                # indices_to_prune = bad_indices.union(nondisc_indices)
                if len(bad_indices) == strain_batch_lls_t.shape[1]:
                    continue
                elif len(bad_indices) > 0:
                    good_indices = [i for i in range(strain_batch_lls_t.shape[1]) if i not in bad_indices]
                    strain_batch_lls_t = strain_batch_lls_t[:, good_indices]

                # ============= Store result.
                self.total_reads += strain_batch_lls_t.shape[1]
                self.batches[t_idx].append(strain_batch_lls_t)

    def advance_epoch(self):
        """
        Allow for callbacks in-between epochs, to enable any intermediary state updates.
        @return:
        """
        pass

    def solve(self,
              num_epochs: int = 1,
              iters: int = 4000,
              num_samples: int = 8000,
              min_lr: float = 1e-4,
              lr_decay_factor: float = 0.25,
              lr_patience: int = 10,
              callbacks: Optional[List[Callable[[int, np.ndarray, float], None]]] = None):
        self.optimize(
            iters=iters,
            num_epochs=num_epochs,
            num_samples=num_samples,
            min_lr=min_lr,
            callbacks=callbacks
        )
        self.posterior.set_parameters(self.optim.params)

    def diagnostic(self, num_importance_samples: int = 10000, batch_size: int = 500):
        pass
        # from chronostrain.util.math import psis_smooth_ratios
        # logger.debug("Running diagnostic...")
        #
        # log_importance_weights = []
        # num_batches = int(np.ceil(num_importance_samples / batch_size))
        # for batch_idx in range(num_batches):
        #     batch_start_idx = batch_idx * batch_size
        #     this_batch_sz = min(num_importance_samples - batch_start_idx, batch_size)
        #     batch_samples = self.posterior.differentiable_sample(num_samples=this_batch_sz).detach()
        #     approx_posterior_ll = self.posterior.log_likelihood(batch_samples).detach()
        #     log_importance_ratios = (
        #             self.model_ll(batch_samples).detach()
        #             + self.data_ll(batch_samples).detach()
        #             - approx_posterior_ll
        #     )
        #     log_importance_weights.append(log_importance_ratios.cpu().numpy())
        #
        # # normalize (for numerical stability).
        # log_importance_weights = np.concatenate(log_importance_weights)
        # log_importance_weights = log_importance_weights - torch.logsumexp(log_importance_weights, dim=0)
        # log_smoothed_weights, k_hat = psis_smooth_ratios(log_importance_weights)
        #
        # logger.debug(f"Estimated Pareto k-hat: {k_hat}")
        # if k_hat > 0.7:
        #     # Extremely large number of samples are needed for stable gradient estimates!
        #     logger.warning(f"Pareto k-hat estimate ({k_hat}) exceeds safe threshold (0.7). "
        #                    "Estimates may be biased/overfit to the variational family. "
        #                    "Perform some empirical testing before proceeding.")

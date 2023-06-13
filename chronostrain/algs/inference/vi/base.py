from abc import abstractmethod, ABC, ABCMeta
from pathlib import Path
from typing import *

import jax
import jax.numpy as np
import numpy as cnp

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.util.math import log_spspmm_exp
from chronostrain.util.optimization import LossOptimizer

from .. import AbstractModelSolver
from .util import divide_columns_into_batches_sparse, log_mm_exp

from chronostrain.logging import create_logger
logger = create_logger(__name__)


_GENERIC_PARAM_TYPE = Dict[str, np.ndarray]
_GENERIC_GRAD_TYPE = _GENERIC_PARAM_TYPE  # the two types usually tend to match.
_GENERIC_SAMPLE_TYPE = Union[Dict[Any, np.ndarray], np.ndarray]


class AbstractPosterior(metaclass=ABCMeta):
    @abstractmethod
    def abundance_sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed, simplex-valued (T x N x S) abundance tensor.
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

    @abstractmethod
    def get_parameters(self) -> _GENERIC_PARAM_TYPE:
        raise NotImplementedError()

    @abstractmethod
    def entropy(self, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
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

    def reparametrize(self, random_samples: _GENERIC_SAMPLE_TYPE, params: _GENERIC_PARAM_TYPE) -> _GENERIC_SAMPLE_TYPE:
        raise NotImplementedError()

    def save(self, path: Path):
        np.savez(
            str(path),
            **self.parameters
        )

    def load(self, path: Path):
        f = np.load(str(path))
        self.parameters = dict(f)


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
                 min_lr: Optional[float] = None,
                 loss_tol: Optional[float] = None,
                 callbacks: Optional[List[Callable[[int, float], None]]] = None):
        time_est = RuntimeEstimator(total_iters=num_epochs, horizon=10)

        logger.info("Starting ELBO optimization.")

        epoch_elbo_prev = -cnp.inf
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
            epoch_elbo_avg = cnp.mean(epoch_elbos).item()

            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, epoch_elbo_avg)

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

            self.optim.scheduler.step(epoch_elbo_avg)
            if min_lr is not None:
                if self.optim.current_learning_rate() <= min_lr:
                    logger.info("Stopping criteria (lr < {}) met after {} epochs.".format(min_lr, epoch))
                    break
            if loss_tol is not None:
                if cnp.abs(epoch_elbo_avg - epoch_elbo_prev) < loss_tol * cnp.abs(epoch_elbo_prev):
                    logger.info("Stopping criteria (Elbo rel. diff < {}) met after {} epochs.".format(loss_tol, epoch))
                    break
                epoch_elbo_prev = epoch_elbo_avg

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
        self.optim.update(elbo_grad)
        return elbo_value


class AbstractADVISolver(AbstractModelSolver, AbstractADVI, ABC):
    """
    A basic implementation of ADVI estimating the posterior p(X|R), with fragments
    F marginalized out.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 optimizer: LossOptimizer,
                 prune_strains: bool,
                 read_batch_size: int = 5000):
        AbstractModelSolver.__init__(
            self,
            model,
            data,
            db
        )
        self.initialize_data(read_batch_size)
        self.prune_reads()
        if prune_strains:
            self.prune_strains_by_read_max()
            # self.prune_strains_by_correlation()
            self.prune_reads()  # do this again to ensure all reads are still useful.
        AbstractADVI.__init__(self, self.create_posterior(), optimizer)

    # noinspection PyAttributeOutsideInit
    def initialize_data(self, read_batch_size):
        logger.debug("Initializing ADVI data structures.")
        if not cfg.model_cfg.use_sparse:
            raise NotImplementedError("ADVI only supports sparse data structures.")

        # (S x R) matrices: Contains P(R = r | S = s) for each read r, strain s.
        self.batches: List[List[np.ndarray]] = [
            [] for _ in range(self.model.num_times())
        ]
        self.total_reads: int = 0

        # Precompute likelihood products.
        logger.debug("Precomputing likelihood marginalization.")
        data_likelihoods = self.data_likelihoods
        for t_idx in range(self.model.num_times()):
            for batch_idx, data_t_batch in enumerate(
                    divide_columns_into_batches_sparse(
                        data_likelihoods.matrices[t_idx],
                        read_batch_size
                    )
            ):
                logger.debug("Precomputing marginalization for t = {}, batch {} ({} reads)".format(
                    t_idx, batch_idx, data_t_batch.shape[1]
                ))
                # ========= Pre-compute likelihood calculations.
                strain_batch_lls_t = log_spspmm_exp(
                    self.model.fragment_frequencies_sparse.T,  # (S x F), note the transpose!
                    data_t_batch  # F x R_batch
                )  # (S x R_batch)

                # ============= Store result.
                self.batches[t_idx].append(strain_batch_lls_t)
                self.total_reads += strain_batch_lls_t.shape[1]

    def prune_reads(self):
        """
        Locate and filter out reads with no good alignments.
        """
        for t in range(self.model.num_times()):
            for batch_idx in range(len(self.batches[t])):
                batch_ll = self.batches[t][batch_idx]

                read_mask = ~np.equal(
                    np.sum(~np.isinf(batch_ll), axis=0),
                    0
                )
                n_good_read_indices = np.sum(read_mask)
                if n_good_read_indices < batch_ll.shape[1]:
                    logger.debug("(t = {}, batch {}) Found {} of {} reads without good alignments.".format(
                        t, batch_idx,
                        batch_ll.shape[1] - n_good_read_indices, batch_ll.shape[1]
                    ))
                self.batches[t][batch_idx] = batch_ll[:, read_mask]

    def prune_strains_by_read_max(self):
        """
        Prune out strains that are not the highest likelihood prior for any read.
        """
        from chronostrain.model import Population

        start_num = self.model.num_strains()
        b = np.zeros(self.model.num_strains())

        for t in range(self.model.num_times()):
            for batch_ll in self.batches[t]:
                batch_ll_max = np.max(batch_ll, axis=0)
                max_hit = np.equal(batch_ll, batch_ll_max[None, :])
                b += np.sum(max_hit, axis=1)
        # good_indices = np.sort(np.argsort(b)[-top_n_to_keep:])
        good_indices, = np.where(b > 0)
        pruned_strains = [self.model.bacteria_pop.strains[i] for i in good_indices]

        # Update data structures
        self.model.bacteria_pop = Population(pruned_strains)
        for t in range(self.model.num_times()):
            for batch_idx in range(len(self.batches[t])):
                batch_ll = self.batches[t][batch_idx]
                self.batches[t][batch_idx] = batch_ll[good_indices, :]

        logger.debug("Pruned {} strains into {} using argmax heuristic.".format(start_num, self.model.num_strains()))

    def prune_strains_by_correlation(self, correlation_threshold: float = 0.99):
        """
        Prune out strains via clustering on the data likelihoods.
        Computes the correlation matrix of the data likelihood values: Corr[p(r|s1), p(r|s2)].
        """
        from chronostrain.model import Population

        S = self.model.bacteria_pop.num_strains()
        dtype = self.batches[0][0].dtype
        corr_min = None
        for t in range(self.model.num_times()):
            first_moment = np.full(S, dtype=dtype, fill_value=-cnp.inf)
            second_moment = np.full((S, S), dtype=dtype, fill_value=-cnp.inf)
            n = 0
            for batch_ll in self.batches[t]:
                first_moment = np.logaddexp(
                    first_moment,
                    jax.nn.logsumexp(batch_ll, axis=-1)
                )
                second_moment = np.logaddexp(
                    second_moment,
                    log_mm_exp(batch_ll, batch_ll.T)
                )
                n += batch_ll.shape[-1]
            first_moment = np.exp(first_moment) / n
            second_moment = np.exp(second_moment) / n
            cov = second_moment - np.outer(first_moment, first_moment)
            del first_moment
            del second_moment
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            del cov
            del std
            corr = np.where(np.isnan(corr), 0., corr)
            corr = np.where(np.isinf(corr), 0., corr)
            if corr_min is None:
                corr_min = corr
            else:
                corr_min = np.minimum(corr_min, corr)

        # perform clustering
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            metric='precomputed',
            linkage='complete',
            distance_threshold=1 - correlation_threshold,
            n_clusters=None
        ).fit(1 - corr_min)

        n_clusters, cluster_labels = clustering.n_clusters_, clustering.labels_
        cluster_representatives = np.array([
            np.where(cluster_labels == c)[0][0]  # Cluster rep is the first by index.
            for c in range(n_clusters)
        ])

        for c in range(n_clusters):
            clust = np.where(cluster_labels == c)[0]
            if len(clust) == 1:
                continue
            clust_members = [self.model.bacteria_pop.strains[i] for i in clust]
            logger.debug("Formed cluster [{}] due to data correlation greater than {}.".format(
                ",".join(f'{s.id}({s.metadata.genus[0]}. {s.metadata.species} {s.name})' for s in clust_members),
                correlation_threshold
            ))

        pruned_strains = [self.model.bacteria_pop.strains[i] for i in np.sort(cluster_representatives)]

        # Update data structures
        self.model.bacteria_pop = Population(pruned_strains)
        for t in range(self.model.num_times()):
            for batch_idx in range(len(self.batches[t])):
                batch_ll = self.batches[t][batch_idx]
                self.batches[t][batch_idx] = batch_ll[cluster_representatives, :]
        logger.debug("Pruned {} strains into {} using correlation-clustering heuristic.".format(S, len(cluster_representatives)))







    # def prune_strains(self, tau=1.0):
    #     """
    #     Summary:
    #     Consider the (S x R) likelihood matrix "A", where R is all reads concatenated, so that A[s,r] = log p(r|s).
    #     Compute the altered matrix B[s,r] = A[s,r] - (max_s A[s,r]).
    #     Compute the row-wise maximum b[s] = max_r B[s,r]
    #     Remove all strains s for which "max_r B[s,r] < log(\tau)", where \tau < 0.
    #
    #     Effectively, all strains s for which ALL reads satisfy "p(r|s) < \tau * p(r|s_best)" is removed.
    #     """
    #     from chronostrain.model import Population
    #
    #     start_num = self.model.num_strains()
    #     log_tau = cnp.log(tau)
    #     b = np.full(self.model.num_strains(), fill_value=-cnp.inf)
    #
    #     for t in range(self.model.num_times()):
    #         for batch_ll in self.batches[t]:
    #             batch_ll_max = np.max(batch_ll, axis=0)
    #             batch_ll_diff = batch_ll - batch_ll_max  # compute matrix "B" for this batch
    #             batch_diff_maxes = np.max(batch_ll_diff, axis=1)  # compute max over reads in this batch
    #             b = np.where(batch_diff_maxes > b, batch_diff_maxes, b)  # update maximum over all reads seen so far
    #     pruned_strains = [
    #         s
    #         for s_idx, s in enumerate(self.model.bacteria_pop.strains)
    #         if b[s_idx] >= log_tau
    #     ]
    #
    #     # Update data structures
    #     self.model.bacteria_pop = Population(pruned_strains)
    #     for t in range(self.model.num_times()):
    #         for batch_idx in range(len(self.batches[t])):
    #             batch_ll = self.batches[t][batch_idx]
    #             self.batches[t][batch_idx] = batch_ll[b >= log_tau, :]
    #
    #     logger.debug("Pruned {} strains into {}.".format(start_num, self.model.num_strains()))

    def advance_epoch(self):
        """
        Allow for callbacks in-between epochs, to enable any intermediary state updates.
        @return:
        """
        pass  # do nothing by default

    @abstractmethod
    def create_posterior(self) -> AbstractReparametrizedPosterior:
        raise NotImplementedError()

    def solve(self,
              num_epochs: int = 1,
              iters: int = 4000,
              num_samples: int = 8000,
              min_lr: Optional[float] = None,
              loss_tol: Optional[float] = None,
              callbacks: Optional[List[Callable[[int, float], None]]] = None):
        self.optimize(
            iters=iters,
            num_epochs=num_epochs,
            num_samples=num_samples,
            min_lr=min_lr,
            loss_tol=loss_tol,
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

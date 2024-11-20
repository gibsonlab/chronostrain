from abc import abstractmethod, ABC
from typing import *
import itertools

import jax
import jax.numpy as jnp
import numpy as cnp

from chronostrain.database import StrainDatabase
from chronostrain.config import cfg
from chronostrain.model import AbundanceGaussianPrior, AbstractErrorModel, TimeSeriesReads, StrainCollection
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.util.math import negbin_fit_frags
from chronostrain.util.optimization import LossOptimizer

from chronostrain.inference.likelihoods import ReadStrainCollectionCache, ReadFragmentMappings
from chronostrain.inference.algs import AbstractModelSolver
from .constants import GENERIC_PARAM_TYPE, GENERIC_SAMPLE_TYPE, GENERIC_GRAD_TYPE
from .posterior import AbstractReparametrizedPosterior
from .marginalizations import load_all_marginalizations

from chronostrain.logging import create_logger
logger = create_logger(__name__)


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
            self.advance_epoch(epoch)

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
            if self.okay_to_terminate():
                if min_lr is not None:
                    if self.optim.current_learning_rate() <= min_lr:
                        logger.info(f"Stopping criteria (lr < {min_lr}) met after {epoch} epochs.")
                        break
                if loss_tol is not None:
                    if cnp.abs(epoch_elbo_avg - epoch_elbo_prev) < loss_tol * cnp.abs(epoch_elbo_prev):
                        logger.info(f"Stopping criteria (Elbo rel. diff < {loss_tol}) met after {epoch} epochs.")
                        break
                    epoch_elbo_prev = epoch_elbo_avg

        # ========== End of optimization
        logger.info("Finished.")

    @abstractmethod
    def advance_epoch(self, epoch: int):
        """
        Do any pre-processing required for a new epoch (e.g. mini-batch read_frags).
        Called at the start of every epoch, including the first one.
        """
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def okay_to_terminate(self):
        """
        Override this if needed.
        Returns "False" if internal state of the posterior model deems it not ready for termination.
        """
        return True

    @abstractmethod
    def elbo_with_grad(
            self,
            params: GENERIC_PARAM_TYPE,
            random_samples: GENERIC_SAMPLE_TYPE
    ) -> Tuple[jnp.ndarray, GENERIC_GRAD_TYPE]:
        """
        :return: The ELBO value, logically separated into `batches` if necessary.
        In implementations, save memory by yielding batches instead of returning a list.
        """
        raise NotImplementedError()

    def optimize_step(
            self,
            random_samples: GENERIC_SAMPLE_TYPE
    ) -> jnp.ndarray:
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
                 gaussian_prior: AbundanceGaussianPrior,
                 error_model: AbstractErrorModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 optimizer: LossOptimizer,
                 prune_strains: bool,
                 read_batch_size: int = 5000,
                 adhoc_corr_threshold: float = 0.99):
        AbstractModelSolver.__init__(
            self,
            gaussian_prior,
            error_model,
            data,
            db
        )

        self.batches, self.paired_batches = self.initialize_marginalizations_cached(read_batch_size)
        self.prune_reads()
        if prune_strains:
            self.prune_strains_by_read_max()
            self.adhoc_clusters = self.prune_strains_by_correlation(correlation_threshold=adhoc_corr_threshold)
            self.prune_reads()  # do this again to ensure all reads are still useful.
        else:
            self.adhoc_clusters = {}
        AbstractADVI.__init__(self, self.create_posterior(), optimizer)

    # noinspection PyAttributeOutsideInit
    def initialize_marginalizations_cached(
            self,
            read_batch_size: int
    ) -> Tuple[List[List[jnp.ndarray]], List[List[jnp.ndarray]]]:
        logger.debug("Initializing ADVI read_frags structures.")
        if not cfg.model_cfg.use_sparse:
            raise NotImplementedError("ADVI only supports sparse read_frags structures.")

        cache = ReadStrainCollectionCache(self.data, self.db, self.gaussian_prior.strain_collection)
        read_likelihoods = ReadFragmentMappings(
            self.data, self.db, self.error_model,
            cache=cache,
            dtype=cfg.engine_cfg.dtype
        ).model_values

        # ============= START: Print some statistics.
        avg_marker_len = int(cnp.median([
            len(m)
            for s in self.db.all_strains()
            for m in s.markers
        ]))
        read_lens = cnp.array([
            len(read)
            for reads_t in self.data
            for read in reads_t
        ])
        if len(read_lens) > 100:
            step = len(read_lens) // 100
            read_lens.sort()
            read_lens = read_lens[::step]

        logger.debug("Read-marker statistics: avg marker = {}, avg read = {}".format(
            avg_marker_len,
            cnp.median(read_lens)
        ))
        # ============= END: Print some statistics.

        # ============= START: Fit negative binomial distribution.
        frag_len_negbin_n, frag_len_negbin_p = negbin_fit_frags(avg_marker_len, read_lens, max_padding_ratio=0.5)
        logger.debug("Negative binomial fit: n={}, p={} (mean={}, std={})".format(
            frag_len_negbin_n,
            frag_len_negbin_p,
            frag_len_negbin_n * (1 - frag_len_negbin_p) / frag_len_negbin_p,
            cnp.sqrt(frag_len_negbin_n * (1 - frag_len_negbin_p)) / frag_len_negbin_p
        ))
        # ============= END: Fit negative binomial distribution.

        return load_all_marginalizations(
            self.data,
            cache,
            read_batch_size,
            frag_len_negbin_n,
            frag_len_negbin_p,
            read_likelihoods
        )

    def prune_reads(self):
        """
        Locate and filter out reads with no good alignments.
        """
        for t in range(self.gaussian_prior.num_times):
            for batch_idx in range(len(self.batches[t])):
                batch_ll = self.batches[t][batch_idx]

                read_mask = ~jnp.equal(
                    jnp.sum(~jnp.isinf(batch_ll), axis=0),
                    0
                )
                n_good_read_indices = jnp.sum(read_mask)
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
        start_num = self.gaussian_prior.num_strains
        b = jnp.zeros(start_num)

        for t in range(self.gaussian_prior.num_times):
            for batch_ll in self.batches[t] + self.paired_batches[t]:
                batch_ll_max = jnp.max(batch_ll, axis=0)
                max_hit = jnp.equal(batch_ll, batch_ll_max[None, :])
                b += jnp.sum(max_hit, axis=1)
        # good_indices = np.sort(np.argsort(b)[-top_n_to_keep:])
        good_indices, = jnp.where(b > 0)
        pruned_strains = [self.gaussian_prior.strain_collection.strains[i] for i in good_indices]

        # Update read_frags structures
        self.gaussian_prior.strain_collection = StrainCollection(
            pruned_strains,
            self.gaussian_prior.strain_collection.db_signature
        )
        for t in range(self.gaussian_prior.num_times):
            for batch_idx in range(len(self.batches[t])):
                batch_ll = self.batches[t][batch_idx]
                self.batches[t][batch_idx] = batch_ll[good_indices, :]
            for batch_idx in range(len(self.paired_batches[t])):
                batch_ll = self.paired_batches[t][batch_idx]
                self.paired_batches[t][batch_idx] = batch_ll[good_indices, :]

        logger.debug("Pruned {} strains into {} using argmax heuristic.".format(
            start_num, self.gaussian_prior.num_strains
        ))

    def prune_strains_by_correlation(self, correlation_threshold: float = 0.99):
        """
        Prune out strains via clustering on the read_frags likelihoods.
        Computes the correlation matrix of the read_frags likelihood values: Corr[p(r|s1), p(r|s2)].
        """
        from chronostrain.model import StrainCollection

        S = self.gaussian_prior.num_strains
        if S == 1:
            s_id = self.gaussian_prior.strain_collection.strains[0].id
            return {s_id: [s_id]}
        if S == 0:
            return {}
        dtype = cfg.engine_cfg.dtype
        corr_min = None
        for t in range(self.gaussian_prior.num_times):
            first_moment = jnp.zeros(S, dtype=dtype)
            second_moment = jnp.zeros((S, S), dtype=dtype)
            n = 0
            for batch_ll in self.batches[t] + self.paired_batches[t]:
                p_normalized = jax.nn.softmax(batch_ll, axis=0)
                first_moment += jnp.sum(p_normalized, axis=1)
                second_moment += p_normalized @ p_normalized.T
                n += batch_ll.shape[-1]
            first_moment = first_moment / n
            second_moment = second_moment / n
            cov = second_moment - jnp.outer(first_moment, first_moment)
            del first_moment
            del second_moment
            std = jnp.sqrt(jnp.diag(cov))
            corr = cov / jnp.outer(std, std)
            del cov
            del std
            corr = jnp.where(jnp.isnan(corr), 0., corr)
            corr = jnp.where(jnp.isinf(corr), 0., corr)
            if corr_min is None:
                corr_min = corr
            else:
                corr_min = jnp.minimum(corr_min, corr)

        # perform clustering
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            metric='precomputed',
            linkage='complete',
            distance_threshold=1 - correlation_threshold,
            n_clusters=None
        ).fit(1 - corr_min)

        n_clusters, cluster_labels = clustering.n_clusters_, clustering.labels_
        cluster_representatives = cnp.array([
            cnp.where(cluster_labels == c)[0][0]  # Cluster rep is the first by index.
            for c in range(n_clusters)
        ])

        adhoc_clusters = {}
        for clust_idx, clust_rep_idx in enumerate(cluster_representatives):
            clust = cnp.where(cluster_labels == clust_idx)[0]

            # Record it into a read_frags structure
            rep_strain = self.gaussian_prior.strain_collection.strains[clust_rep_idx]
            adhoc_clusters[rep_strain.id] = [
                self.gaussian_prior.strain_collection.strains[c_idx].id
                for c_idx in clust
            ]

            if len(clust) > 1:
                clust_members = [self.gaussian_prior.strain_collection.strains[i] for i in clust]
                avg_corr = cnp.mean([
                    corr[c_idx_i, c_idx_j]
                    for c_idx_i, c_idx_j in itertools.combinations(clust, r=2)
                ])
                logger.debug("Formed cluster [{}] due to read_frags correlation greater than {} (AVG_CORR={}).".format(
                    ",".join(f'{s.id}({s.metadata.genus[0]}. {s.metadata.species} {s.name})' for s in clust_members),
                    correlation_threshold,
                    avg_corr
                ))

        # Update read_frags structures
        cluster_representatives = cnp.sort(cluster_representatives)
        self.gaussian_prior.strain_collection = StrainCollection(
            [self.gaussian_prior.strain_collection.strains[i] for i in cluster_representatives],
            self.gaussian_prior.strain_collection.db_signature
        )
        for t in range(self.gaussian_prior.num_times):
            for batch_idx in range(len(self.batches[t])):
                batch_ll = self.batches[t][batch_idx]
                self.batches[t][batch_idx] = batch_ll[cluster_representatives, :]
            for batch_idx in range(len(self.paired_batches[t])):
                batch_ll = self.paired_batches[t][batch_idx]
                self.paired_batches[t][batch_idx] = batch_ll[cluster_representatives, :]
        logger.debug("Pruned {} strains into {} using correlation-clustering heuristic.".format(S, len(cluster_representatives)))
        return adhoc_clusters

    def advance_epoch(self, epoch):
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
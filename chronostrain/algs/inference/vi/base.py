from abc import abstractmethod, ABC, ABCMeta
from pathlib import Path
from typing import Optional, List, Callable, Iterator, Type, Dict, Any

import numpy as np
import torch
from torch.nn import Parameter

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.util.math.matrices import *

from .. import AbstractModelSolver
from .util import divide_columns_into_batches_sparse

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class AbstractPosterior(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed, simplex-valued (T x N x S) abundance tensor.
        """
        pass

    @abstractmethod
    def mean(self) -> torch.Tensor:
        """
        Returns the mean of this posterior distribution.
        :return: A time-indexed (T x S) abundance tensor.
        """
        pass

    @abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> float:
        pass

    @abstractmethod
    def save(self, target_path: Path):
        pass


class AbstractReparametrizedPosterior(AbstractPosterior, ABC):
    def log_likelihood(self, samples: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def trainable_parameters(self) -> List[Parameter]:
        raise NotImplementedError()

    @abstractmethod
    def mean(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def differentiable_sample(self, num_samples: int) -> torch.Tensor:
        """
        Return reparametrized (differentiable) samples, to be used for autograd.
        """
        raise NotImplementedError()


class AbstractADVI(ABC):
    """
    An abstraction of the autograd-driven (black-box) VI implementation.
    """

    def __init__(self, posterior: AbstractReparametrizedPosterior, device: torch.device):
        self.posterior = posterior
        self.device = device

    def optimize(self,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,
                 num_epochs: int = 1,
                 iters: int = 50,
                 num_samples: int = 150,
                 min_lr: float = 1e-4,
                 callbacks: Optional[List[Callable[[int, torch.Tensor, float], None]]] = None):
        time_est = RuntimeEstimator(total_iters=num_epochs, horizon=10)
        reparam_samples = None
        elbo_value = 0.0

        logger.info("Starting ELBO optimization.")
        for epoch in range(1, num_epochs + 1, 1):
            self.advance_epoch()
            epoch_elbos = []
            time_est.stopwatch_click()
            for it in range(1, iters + 1, 1):
                reparam_samples = self.posterior.differentiable_sample(
                    num_samples=num_samples
                )

                elbo_value = self.optimize_step(reparam_samples, optimizer)
                epoch_elbos.append(elbo_value)

            # ===========  End of epoch
            epoch_elbo_avg = np.mean(epoch_elbos)

            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, reparam_samples, elbo_value)

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            logger.info(
                "Epoch {epoch} | time left: {t:.2f} min. | Average ELBO = {elbo:.2f} | LR = {lr}".format(
                    epoch=epoch,
                    t=time_est.time_left() / 60000,
                    elbo=epoch_elbo_avg,
                    lr=optimizer.param_groups[-1]['lr']
                )
            )

            lr_scheduler.step(-epoch_elbo_avg)
            if optimizer.param_groups[-1]['lr'] < min_lr:
                logger.info("Stopping criteria met after {} epochs.".format(epoch))
                break

        # ========== End of optimization
        logger.info("Finished.")

        if self.device == torch.device("cuda"):
            logger.info(
                "ADVI CUDA memory -- [MaxAlloc: {} MiB]".format(
                    torch.cuda.max_memory_allocated(self.device) / 1048576
                )
            )
        else:
            logger.debug(
                "ADVI CPU memory usage -- [Not implemented]"
            )

    @abstractmethod
    def advance_epoch(self):
        """
        Do any pre-processing required for a new epoch (e.g. mini-batch data).
        Called at the start of every epoch, including the first one.
        """
        pass

    @abstractmethod
    def elbo(self, samples: torch.Tensor) -> Iterator[torch.Tensor]:
        """
        :return: The ELBO value, logically separated into `batches` if necessary.
        In implementations, save memory by yielding batches instead of returning a list.
        """
        pass

    def optimize_step(self, samples: torch.Tensor,
                      optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad(set_to_none=True)
        elbo_value = 0.0

        # Accumulate overall gradient estimator in batches.
        for elbo_chunk in self.elbo(samples):
            # Save float value for callbacks.
            elbo_value += elbo_chunk.item()

            # Gradient accumulation: Maximize ELBO by minimizing (-ELBO) over this chunk.
            elbo_loss_chunk = -elbo_chunk
            elbo_loss_chunk.backward(retain_graph=True)

        # Use the accumulated gradient to update.
        optimizer.step()
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
                 posterior: AbstractReparametrizedPosterior,
                 read_batch_size: int = 5000):
        AbstractModelSolver.__init__(
            self,
            model,
            data,
            db
        )
        self.read_batch_size = read_batch_size

        AbstractADVI.__init__(
            self,
            posterior,
            device=cfg.torch_cfg.device
        )

        logger.debug("Initializing ADVI data structures.")
        if not cfg.model_cfg.use_sparse:
            raise NotImplementedError("ADVI only supports sparse data structures.")

        # (S x R) matrices: Contains P(R = r | S = s) for each read r, strain s.
        self.batches: List[List[torch.Tensor]] = [
            [] for _ in range(model.num_times())
        ]
        self.total_reads: int = 0

        # Precompute this (only possible in V1).
        logger.debug("Precomputing likelihood products.")
        data_likelihoods = self.data_likelihoods
        for t_idx in range(model.num_times()):
            data_ll_t = data_likelihoods.matrices[t_idx]  # F_ x R, sparse CPU
            projector = data_likelihoods.projectors[t_idx]  # sparse CPU

            for batch_idx, data_t_batch in enumerate(divide_columns_into_batches_sparse(data_ll_t, self.read_batch_size)):
                # ========= Pre-compute likelihood calculations.
                strain_batch_lls_t = log_spspmm_exp(
                    ColumnSectionedSparseMatrix.from_sparse_matrix(
                        projector.sparse_mul(self.model.fragment_frequencies_sparse).t()
                    ),  # (S x F_), note the transpose!
                    RowSectionedSparseMatrix.from_sparse_matrix(data_t_batch)  # (F_ x R_batch)
                )  # (S x R_batch)

                # ========= Locate and filter out reads with no good alignments.
                bad_indices = {
                    int(x)
                    for x in torch.where(
                        torch.eq(
                            torch.sum(~torch.isinf(strain_batch_lls_t), dim=0).cpu(),
                            torch.tensor(0)
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
                self.batches[t_idx].append(strain_batch_lls_t.to(cfg.torch_cfg.device))

    def advance_epoch(self):
        """
        Allow for callbacks in-between epochs, to enable any intermediary state updates.
        @return:
        """
        pass

    def solve(self,
              optimizer_class: Type[torch.optim.Optimizer],
              optimizer_args: Dict[str, Any],
              num_epochs: int = 1,
              iters: int = 4000,
              num_samples: int = 8000,
              min_lr: float = 1e-4,
              lr_decay_factor: float = 0.25,
              lr_patience: int = 10,
              callbacks: Optional[List[Callable[[int, torch.Tensor, float], None]]] = None):
        optimizer_args['params'] = self.posterior.trainable_parameters()
        optimizer = optimizer_class(**optimizer_args)

        logger.debug("LR scheduler parameters: decay={}, patience={}".format(
            lr_decay_factor,
            lr_patience
        ))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_decay_factor,
            patience=lr_patience,
            threshold=1e-4,
            threshold_mode='rel',
            mode='min'  # track (-ELBO) and decrease LR when it stops decreasing.
        )

        self.optimize(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iters=iters,
            num_epochs=num_epochs,
            num_samples=num_samples,
            min_lr=min_lr,
            callbacks=callbacks
        )

    @abstractmethod
    def data_ll(self, samples: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def model_ll(self, samples: torch.Tensor) -> torch.Tensor:
        pass

    def diagnostic(self, num_importance_samples: int = 10000, batch_size: int = 500):
        from chronostrain.util.math import psis_smooth_ratios
        logger.debug("Running diagnostic...")

        log_importance_weights = []
        num_batches = int(np.ceil(num_importance_samples / batch_size))
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * batch_size
            this_batch_sz = min(num_importance_samples - batch_start_idx, batch_size)
            batch_samples = self.posterior.differentiable_sample(num_samples=this_batch_sz).detach()
            approx_posterior_ll = self.posterior.log_likelihood(batch_samples).detach()
            log_importance_ratios = (
                    self.model_ll(batch_samples).detach()
                    + self.data_ll(batch_samples).detach()
                    - approx_posterior_ll
            )
            log_importance_weights.append(log_importance_ratios.cpu().numpy())

        # normalize (for numerical stability).
        log_importance_weights = np.concatenate(log_importance_weights)
        log_importance_weights = log_importance_weights - torch.logsumexp(log_importance_weights, dim=0)
        log_smoothed_weights, k_hat = psis_smooth_ratios(log_importance_weights)

        logger.debug(f"Estimated Pareto k-hat: {k_hat}")
        if k_hat > 0.7:
            # Extremely large number of samples are needed for stable gradient estimates!
            logger.warning(f"Pareto k-hat estimate ({k_hat}) exceeds safe threshold (0.7). "
                           "Estimates may be biased/overfit to the variational family. "
                           "Perform some empirical testing before proceeding.")

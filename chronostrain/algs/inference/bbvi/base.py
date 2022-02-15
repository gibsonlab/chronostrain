from abc import abstractmethod, ABC
from typing import Optional, List, Callable, Iterator

import torch
from torch.nn import Parameter

from chronostrain import create_logger
from chronostrain.algs.inference.bbvi.posteriors.base import AbstractReparametrizedPosterior
from chronostrain.util.benchmarking import RuntimeEstimator

logger = create_logger(__name__)


class AbstractBBVI(ABC):
    """
    An abstraction of a black-box VI implementation.
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
            epoch_elbo = 0.0
            time_est.stopwatch_click()
            for it in range(1, iters + 1, 1):
                reparam_samples = self.posterior.reparametrized_sample(
                    num_samples=num_samples
                )
                reparam_lls = self.posterior.reparametrized_sample_log_likelihoods(reparam_samples)

                elbo_value = self.optimize_step(reparam_samples, reparam_lls, optimizer)
                epoch_elbo += elbo_value

            # ===========  End of epoch
            epoch_elbo /= iters
            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, reparam_samples, elbo_value)

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            logger.info(
                "Epoch {epoch} | time left: {t:.2f} min. | Average ELBO = {elbo:.2f} | LR = {lr}".format(
                    epoch=epoch,
                    t=time_est.time_left() / 60000,
                    elbo=epoch_elbo,
                    lr=optimizer.param_groups[-1]['lr']
                )
            )

            lr_scheduler.step(epoch_elbo)
            if optimizer.param_groups[-1]['lr'] < min_lr:
                logger.info("Stopping criteria met after {} epochs.".format(epoch))
                break

        # ========== End of optimization
        logger.info("Finished.")

        if self.device == torch.device("cuda"):
            logger.info(
                "BBVI CUDA memory -- [MaxAlloc: {} MiB]".format(
                    torch.cuda.max_memory_allocated(self.device) / 1048576
                )
            )
        else:
            logger.debug(
                "BBVI CPU memory usage -- [Not implemented]"
            )

    @property
    def trainable_params(self) -> List[Parameter]:
        return self.posterior.trainable_parameters()

    @abstractmethod
    def elbo(self, samples: torch.Tensor, posterior_sample_lls: torch.Tensor) -> Iterator[torch.Tensor]:
        """
        :return: The ELBO value, logically separated into `batches` if necessary.
        In implementations, save memory by yielding batches instead of returning a list.
        """
        pass

    def optimize_step(self, samples: torch.Tensor,
                      posterior_sample_lls: torch.Tensor,
                      optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()

        elbo_value = 0.0
        for elbo_chunk in self.elbo(samples, posterior_sample_lls):
            # Maximize ELBO by minimizing (-ELBO).
            elbo_loss_chunk = -elbo_chunk
            elbo_loss_chunk.backward(retain_graph=True)

            # Save float value for callbacks.
            elbo_value += elbo_chunk.item()
        optimizer.step()
        return elbo_value

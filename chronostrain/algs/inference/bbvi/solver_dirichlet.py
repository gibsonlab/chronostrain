# from typing import Iterator, Optional
#
# import numpy as np
# import torch
#
# from chronostrain.database import StrainDatabase
# from chronostrain.model.generative import GenerativeModel
# from chronostrain.model.io import TimeSeriesReads
#
# from chronostrain.logging import create_logger
# from chronostrain.util.math.matrices import log_mm_exp
# from .base import AbstractADVISolver
# from ...subroutines.likelihoods import DataLikelihoods
# from .posteriors import ReparametrizedDirichletPosterior
#
# logger = create_logger(__name__)
#
#
# class ADVIDirichletSolver(AbstractADVISolver):
#     """
#     A basic implementation of ADVI estimating the posterior p(X|R), with fragments
#     F marginalized out.
#     """
#     def __init__(self,
#                  model: GenerativeModel,
#                  data: TimeSeriesReads,
#                  db: StrainDatabase,
#                  read_batch_size: int = 5000,
#                  num_cores: int = 1,
#                  precomputed_data_likelihoods: Optional[DataLikelihoods] = None):
#         logger.info("Initializing solver with Dirichlet posterior (q(Y) on S-1 degrees of freedom)")
#         posterior = ReparametrizedDirichletPosterior(model.num_strains(), model.num_times())
#         super().__init__(
#             model=model,
#             data=data,
#             db=db,
#             posterior=posterior,
#             read_batch_size=read_batch_size,
#             num_cores=num_cores,
#             precomputed_data_likelihoods=precomputed_data_likelihoods
#         )
#
#     def elbo(self,
#              log_dirichlet_samples: torch.Tensor
#              ) -> Iterator[torch.Tensor]:
#         """
#         Computes the ADVI approximation to the ELBO objective, holding the read-to-fragment posteriors
#         fixed. The entropic term is computed in closed-form, while the cross-entropy is given a monte-carlo estimate.
#
#         The formula is:
#             ELBO = E_Q(log P - log Q)
#                 = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))
#                 = E_{X~Q}(log P(X) + P(R|X)) + H(Q)
#
#         :param log_dirichlet_samples: A (T x N x S) tensor, where T = # of timepoints, N = # of samples, S = # of strains.
#         :return: An estimate of the ELBO, using the provided samples via the above formula.
#         """
#
#         """
#         ELBO original formula:
#             E_Q[P(X)] - E_Q[Q(X)] + E_{F ~ phi} [log P(F | X)]
#
#         To save memory on larger frag spaces, split the ELBO up into several pieces.
#         """
#         # # ======== H(Q) = E_Q[-log Q(X)]
#         # yield self.posterior.entropy()
#         #
#         # # ======== E[log P(X)]
#         # yield torch.mean(self.model_ll_with_grad(log_dirichlet_samples))
#
#         # ======== E[log P(R|X)] = E[log Σ_S P(R|S)P(S|X)]
#         for t_idx in np.random.permutation(self.model.num_times()):
#             for batch_lls in self.batches[t_idx]:
#                 batch_wt = batch_lls.shape[1] / self.total_reads
#                 entropy = self.posterior.entropy()
#                 prior = torch.mean(self.model_ll_with_grad(log_dirichlet_samples))
#
#                 # Average of (N x R_batch) entries, we only want to divide by 1/N and not 1/(N*R_batch)
#                 data_ll = batch_lls.shape[1] * torch.mean(log_mm_exp(log_dirichlet_samples[t_idx], batch_lls))
#                 yield data_ll + batch_wt * (entropy + prior)
#
#     def data_ll(self, log_dirichlet_samples: torch.Tensor) -> torch.Tensor:
#         ans = torch.zeros(size=(log_dirichlet_samples.shape[1],), device=log_dirichlet_samples.device)
#         for t_idx in range(self.model.num_times()):
#             for batch_lls in self.batches[t_idx]:
#                 batch_matrix = log_mm_exp(log_dirichlet_samples[t_idx], batch_lls).detach()
#                 ans = ans + torch.sum(batch_matrix, dim=1)
#         return ans
#
#     def model_ll_with_grad(self, log_dirichlet_samples: torch.Tensor) -> torch.Tensor:
#         x = log_dirichlet_samples[:, :, :-1] - log_dirichlet_samples[:, :, -1].unsqueeze(2)
#         return self.model.log_likelihood_x(X=x)
#
#     def model_ll(self, log_dirichlet_samples: torch.Tensor) -> torch.Tensor:
#         return self.model_ll_with_grad(log_dirichlet_samples.detach())

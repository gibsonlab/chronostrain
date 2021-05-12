"""
  bbvi.py (pytorch implementation)
  Black-box Variational Inference
  Author: Younhun Kim

  This is an implementation of BBVI for the posterior q(X_1) q(X_2 | X_1) ...
  (Note: doesn't work as well as BBVI for mean-field assumption.)
"""
from typing import Iterable, Tuple, Optional, Callable, Dict, Union

import torch_sparse
from torch.nn.functional import softplus
from torch.distributions import Normal

from chronostrain.config import cfg
from chronostrain.algs.vi import AbstractPosterior
from chronostrain.model import GenerativeModel, Fragment
from chronostrain.model.io import TimeSeriesReads
from chronostrain.algs.base import AbstractModelSolver
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.util.math import *
from . import logger


# ============== JIT helpers
from ..util.sparse import normalize_sparse_2d


# @torch.jit.script
# def coordinate_sum2d(sparse_indices: torch.Tensor,
#                      sparse_values: torch.Tensor,
#                      dim: int,
#                      idx: int) -> float:
#     """
#     For the specified sparse 2-d matrix (say of size R x C),
#     computes the sum of the specified row- or column- vector (given by `idx`).
#     (To be precise, the dense version of this would be torch.sum(x[idx, :]) if dim == 0,
#      or torch.sum(x[:, idx]) if dim == 1.)
# 
#     Note: `dim` does not mean that the sum occurs across specified dimension, it refers to which dimension
#         to use to index (`idx`). e.g. If dim = 0, idx = r calculates the sum of the r-th row vector.
#     """
#     return sparse_values[sparse_indices[dim] == idx].sum()
# 
# 
# @torch.jit.script
def sparse_slice(x: torch.Tensor, dim: int, idx: int):
    """
    Returns the dimension `dim`, index `idx` slice of the sparse matrix, as a dense 1-d tensor.
    Equivalent to x[idx, :] if dim == 0, or x[:, idx] if dim == 1.
    :param x:
    :param dim:
    :param idx:
    :return:
    """
    matching_entries = torch.where(x.indices()[dim, :] == idx)[0]
    return torch.sparse_coo_tensor(
        indices=x.indices()[1 - dim, matching_entries].unsqueeze(0),
        values=x.values()[matching_entries],
        size=[x.size()[1 - dim]]
    )
# 
# 
# @torch.jit.script
# def sparse_sum(indices: torch.Tensor, values: torch.Tensor, dim: int, rows: int, cols: int):
#     x = torch.sparse_coo_tensor(indices, values, (rows, cols)).coalesce()
#     return torch.sparse.sum(x, dim=dim).to_dense()
# 
# 
# # @torch.jit.script
# def vi_log_expectation(phi_sparse,
#                        x: torch.Tensor,
#                        frag_likelihoods_sparse: torch.Tensor):
#     """
#     JIT-compiled helper for elbo_marginal_gaussian_sparse.
# 
#     The log-likelihood expectations for each sample X (length-N array, where N = # of samples.):
#         E_{Frag \sim Phi}[ P(Frag | X) ]
#         = Sum_r Sum_f log(softmax(X) @ frag_likelihoods[f, :]) * phi[f,r]
# 
#     Note: This is naively a O(|R| * |F|) computation, but by switching the order of summation into "Sum_f Sum_r",
#      this reduces to an O(|R| + |F|) computation.
#      The only issue is that Sum_r phi[f,r] is not memory-efficient (phi is only column-sparse), so some
#      JIT optimizations must be made in evaluating the outer sum (to compensate for not being able to vectorize).
# 
#     :param phi_sparse: The posterior fragment likelihoods (phi), an (F x R) sparse matrix.
#     :param x: (N x S) matrix.
#     :param frag_likelihoods_indices: The indices of `frag_likelihoods`, a (F x S) sparse matrix.
#     :param frag_likelihoods_values: The values of `frag_likelihoods`, a (F x S) sparse matrix.
#     :param num_fragments: the value of F, the number of fragments.
#     :return: The log-likelihood expectations for each sample (length-N array).
#     """
#     answer = torch.zeros(size=[x.size()[0]], dtype=x.dtype, device=x.device)  # Length N
#     softmax_x = torch.softmax(x, dim=1)
#     num_strains = x.size()[1]
# 
#     phi_sum = torch.sparse.sum(phi_sparse, dim=1)
# 
#     # These are all dense operations.
#     phi_sum * torch.log(
#         softmax_x @ frag_likelihoods_sparse.t()
#     )
# 
#     # for f in range(num_fragments):
#     #     phi_sum_f = coordinate_sum2d(phi_indices, phi_values, 0, f)
#     #     frag_likelihoods_slice_f = dense_slice(
#     #         frag_likelihoods_indices,
#     #         frag_likelihoods_values,
#     #         0,
#     #         f,
#     #         num_fragments,
#     #         num_strains
#     #     )  # length S
#     #
#     #     answer += phi_sum_f * torch.log(
#     #         torch.mv(
#     #             softmax_x,  # (N x S)
#     #             frag_likelihoods_slice_f  # length S
#     #         )  # length N
#     #     )
#     # return answer


# ============== Posteriors

class PositiveScaleLayer(torch.nn.Module):

    def __init__(self, size: torch.Size):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(size))

    def forward(self, input: torch.Tensor):
        return softplus(self.scale) * input


class GaussianPosteriorFullCorrelation(AbstractPosterior):
    def __init__(self, model: GenerativeModel):
        """
        Mean-field assumption:
        1) Parametrize the (T x S) trajectory as a (TS)-dimensional Gaussian.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param model: The generative model to use.
        """
        # Check: might need this to be a matrix, not a vector.
        self.model = model

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_network = torch.nn.Linear(
            in_features=self.model.num_times() * self.model.num_strains(),
            out_features=self.model.num_times() * self.model.num_strains()
        ).to(cfg.torch_cfg.device)
        torch.nn.init.eye(self.reparam_network.weight)
        self.trainable_parameters = self.reparam_network.parameters()
        self.standard_normal = Normal(loc=0.0, scale=1.0)

    def sample(self, num_samples=1) -> torch.Tensor:
        return self.reparametrized_sample(
            num_samples=num_samples, output_log_likelihoods=False, detach_grad=True
        )

    def reparametrized_sample(self,
                              num_samples=1,
                              output_log_likelihoods=False,
                              detach_grad=False
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(num_samples, self.model.num_times() * self.model.num_strains())
        )

        # ======= Reparametrization
        reparametrized_samples = self.reparam_network(std_gaussian_samples)

        if detach_grad:
            reparametrized_samples = reparametrized_samples.detach()

        if output_log_likelihoods:
            log_likelihoods = torch.distributions.MultivariateNormal(
                loc=self.reparam_network.bias,
                covariance_matrix=self.reparam_network.weight.t().mm(self.reparam_network.weight)
            ).log_prob(reparametrized_samples)
            return reparametrized_samples.view(
                num_samples, self.model.num_times(), self.model.num_strains()
            ).transpose(0, 1), log_likelihoods
        else:
            return reparametrized_samples.view(
                num_samples, self.model.num_times(), self.model.num_strains()
            ).transpose(0, 1)


class GaussianPosteriorStrainCorrelation(AbstractPosterior):
    def __init__(self, model: GenerativeModel):
        """
        Mean-field assumption:
        1) Parametrize X_1, ..., X_T as independent S-dimensional gaussians.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param model: The generative model to use.
        """
        # Check: might need this to be a matrix, not a vector.
        self.model = model

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks = []

        for t_idx in range(self.model.num_times()):
            linear_layer = torch.nn.Linear(
                in_features=self.model.num_strains(),
                out_features=self.model.num_strains(),
                bias=True,
            ).to(cfg.torch_cfg.device)
            torch.nn.init.eye_(linear_layer.weight)
            self.reparam_networks.append(linear_layer)

        self.trainable_parameters = []
        for network in self.reparam_networks:
            self.trainable_parameters += network.parameters()

        self.standard_normal = Normal(loc=0.0, scale=1.0)

    def sample(self, num_samples=1) -> torch.Tensor:
        return self.reparametrized_sample(
            num_samples=num_samples, output_log_likelihoods=False, detach_grad=True
        )

    def reparametrized_sample(self,
                              num_samples=1,
                              output_log_likelihoods=False,
                              detach_grad=False
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.model.num_times(), num_samples, self.model.num_strains())
        )

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[t].forward(std_gaussian_samples[t, :, :])
            for t in range(self.model.num_times())
        ], dim=0)  # (T x N x S)

        if detach_grad:
            samples = samples.detach()

        if output_log_likelihoods:
            return samples, self.log_likelihood(samples)
        else:
            return samples

    def log_likelihood(self, samples):
        # samples is (T x N x S)
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True)
        for t in range(self.model.num_times()):
            samples_t = samples[t, :, :]
            linear = self.reparam_networks[t]
            log_likelihood_t = torch.distributions.MultivariateNormal(
                loc=linear.bias,
                covariance_matrix=linear.weight.t().mm(linear.weight)
            ).log_prob(samples_t)
            ans = ans + log_likelihood_t
        return ans


class GaussianPosteriorTimeCorrelation(AbstractPosterior):
    def __init__(self, model: GenerativeModel):
        """
        Mean-field assumption:
        1) Parametrize X_1, X_2, ..., X_S as independent T-dimensional gaussians (one per strain).
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param model: The generative model to use.
        """
        # Check: might need this to be a matrix, not a vector.
        self.model = model

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks: Dict[int, torch.nn.Module] = dict()

        for s_idx in range(self.model.num_strains()):
            linear_layer = torch.nn.Linear(
                in_features=self.model.num_times(),
                out_features=self.model.num_times(),
                bias=True,
            ).to(cfg.torch_cfg.device)
            torch.nn.init.eye(linear_layer.weight)
            self.reparam_networks[s_idx] = linear_layer
        self.trainable_parameters = []
        for network in self.reparam_networks.values():
            self.trainable_parameters += network.parameters()

        self.standard_normal = Normal(loc=0.0, scale=1.0)

    def sample(self, num_samples=1) -> torch.Tensor:
        return self.reparametrized_sample(
            num_samples=num_samples, output_log_likelihoods=False, detach_grad=True
        )

    def reparametrized_sample(self,
                              num_samples=1,
                              output_log_likelihoods=False,
                              detach_grad=False
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.model.num_strains(), num_samples, self.model.num_times())
        )

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[s_idx].forward(std_gaussian_samples[s_idx, :, :])
            for s_idx in range(self.model.num_strains())
        ], dim=0)

        if detach_grad:
            samples = samples.detach()

        if output_log_likelihoods:
            return samples.transpose(0, 2), self.log_likelihoods(samples)
        else:
            return samples.transpose(0, 2)

    def log_likelihoods(self, samples):
        # For this posterior, samples is (S x N x T).
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True)
        for s in range(self.model.num_strains()):
            samples_s = samples[s, :, :]
            linear = self.reparam_networks[s]
            log_likelihood_s = torch.distributions.MultivariateNormal(
                loc=linear.bias,
                covariance_matrix=linear.weight.t().mm(linear.weight)
            ).log_prob(samples_s)
            ans = ans + log_likelihood_s
        return ans


class FragmentPosterior(object):
    def __init__(self, model: GenerativeModel):
        self.model = model

        # length-T list of (F x N_t) tensors.
        self.phi: List[torch.Tensor] = []

    def top_fragments(self, time_idx, read_idx, top=5) -> Iterable[Tuple[Fragment, float]]:
        phi_t = self.phi[time_idx]
        if phi_t.is_sparse:
            # Assumes that this is a 1-d tensor (representing sparse valeus)
            read_slice = sparse_slice(phi_t, 1, read_idx).coalesce()

            sparse_topk = torch.topk(
                input=read_slice.values(),
                k=min(top, len(read_slice.values())),
                sorted=True
            )
            for sparse_idx, frag_prob in zip(sparse_topk.indices, sparse_topk.values):
                frag_idx = read_slice.indices()[0, sparse_idx]
                yield self.model.get_fragment_space().get_fragment_by_index(frag_idx), frag_prob.item()
        else:
            topk_result = torch.topk(
                input=phi_t[:, read_idx],
                k=top,
                sorted=True
            )
            for frag_idx, frag_prob in zip(topk_result.indices, topk_result.values):
                yield self.model.get_fragment_space().get_fragment_by_index(frag_idx), frag_prob.item()


class BBVISolver(AbstractModelSolver):
    """
    An abstraction of a black-box VI implementation.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 correlation_type: str = "time"):
        super().__init__(model, data)
        self.correlation_type = correlation_type
        if correlation_type == "time":
            self.gaussian_posterior = GaussianPosteriorTimeCorrelation(model=model)
        elif correlation_type == "strain":
            self.gaussian_posterior = GaussianPosteriorStrainCorrelation(model=model)
        elif correlation_type == "full":
            self.gaussian_posterior = GaussianPosteriorFullCorrelation(model=model)
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        self.fragment_posterior = FragmentPosterior(model=model)  # time-indexed list of F x N tensors.

    def elbo_marginal_gaussian(self,
                               x_samples: torch.Tensor,
                               posterior_gaussian_log_likelihoods: torch.Tensor) -> torch.Tensor:
        """
        Computes the monte-carlo approximation to the ELBO objective, holding the read-to-fragment posteriors fixed.

        The formula is (by mean field assumption):
            ELBO = E_Q(log P - log Q)
                = E_{X~Qx}(log P(X)) + E_{X~Qx,F~Qf}(log P(F|X))
                    + E_{F~Qf}(log P(R|F)) - E_{X~Qx}(log Qx(X)) - E_{F~Qf}(log Qf(F))

        Since the purpose is to compute gradients, we leave out the third and fifth terms (constant with respect to
        the Gaussian parameters).
        Replacing E_Qx with empirical samples (replacing X with Xi), it becomes:

                = E_Qx(log P(X)) + E_{X~Qx,F~Qf}(log P(F|X)) - E_Qx(log Qx(Xi))
                = MEAN[ log P(Xi) + E_{F ~ Qf}(log P(F|Xi)) - log Qx(Xi) ]

        (With re-parametrization trick X = mu + Sigma^(1/2) * EPS, so that the expectations do not depend on mu/sigma.)

        :param x_samples: A (T x N x S) tensor, where T = # of timepoints, N = # of samples, S = # of strains.
        :param posterior_gaussian_log_likelihoods: A length-N (one-dimensional) tensor of the joint log-likelihood
            each (T x S) slice.
        :return: An estimate of the ELBO, using the provided samples via the above formula.
        """
        if cfg.model_cfg.use_sparse:
            return self.elbo_marginal_gaussian_sparse(x_samples, posterior_gaussian_log_likelihoods)
        else:
            return self.elbo_marginal_gaussian_dense(x_samples, posterior_gaussian_log_likelihoods)

    def elbo_marginal_gaussian_dense(self,
                                     x_samples: torch.Tensor,
                                     posterior_gaussian_log_likelihoods: torch.Tensor) -> torch.Tensor:
        # ======== log P(Xi)
        model_gaussian_log_likelihoods = self.model.log_likelihood_x(X=x_samples)

        # ======== E_{F ~ Qf}(log P(F|Xi))
        n_samples = x_samples.size()[1]
        expectation_model_log_fragment_probs = torch.zeros(
            size=(n_samples,),
            dtype=cfg.torch_cfg.default_dtype,
            device=cfg.torch_cfg.device
        )
        for t_idx in range(self.model.num_times()):
            """
            Expectation is \sum_{read} \sum_{frag} phi[read,frag] * logP(frag|X).
            For a speedup, we switch order of summation: \sum_{frag} logP(frag|X) * (\sum_{read} phi[read,frag])
            """
            model_frag_likelihoods_t = softmax(
                x_samples[t_idx, :, :],  # (N x S)
                dim=1
            ).mm(
                self.model.get_fragment_frequencies().t()  # (S x F)
            ).log()  # (N x F)
            expectation_model_log_fragment_probs += model_frag_likelihoods_t.mv(
                self.fragment_posterior.phi[t_idx].sum(dim=1)  # length F
            )  # length N

        elbo_samples = (model_gaussian_log_likelihoods
                        + expectation_model_log_fragment_probs
                        - posterior_gaussian_log_likelihoods)
        return elbo_samples.mean()

    def elbo_marginal_gaussian_sparse(self, x_samples: torch.Tensor, posterior_gaussian_log_likelihoods: torch.Tensor) -> torch.Tensor:
        """
        Same as dense implementation, but computes the middle term E_{F ~ Qf}(log P(F|Xi)) using monte-carlo samples of F from phi.
        :param x_samples:
        :param posterior_gaussian_log_likelihoods:
        :return:
        """
        # ======== log P(Xi)
        model_gaussian_log_likelihoods = self.model.log_likelihood_x(X=x_samples)

        # ======== E_{F ~ Qf}(log P(F|Xi))
        n_samples = x_samples.size()[1]
        expectation_model_log_fragment_probs = torch.zeros(
            size=(n_samples,),
            dtype=cfg.torch_cfg.default_dtype,
            device=cfg.torch_cfg.device
        )
        for t_idx in range(self.model.num_times()):
            softmax_x_t = torch.softmax(x_samples[t_idx, :, :], dim=1)  # (N x S)
            phi_sum = torch.sparse.sum(self.fragment_posterior.phi[t_idx], dim=1).to_dense()  # (length F)

            # y_ = torch.log(
            #     # (F x S) sparse matrix @ (S x N) dense
            #     torch_sparse.spmm(
            #         self.model.get_fragment_frequencies().indices(),
            #         self.model.get_fragment_frequencies().values(),
            #         self.model.get_fragment_frequencies().size()[0],
            #         self.model.get_fragment_frequencies().size()[1],
            #         softmax_x_t.t()
            #     ).t()
            #     # softmax_x_t @ self.model.get_fragment_frequencies().t()  # (N x S) @ (S x F) = (N x F)
            # )
            # print(y_)
            # print(phi_sum)
            # exit(1)

            # These are all dense operations.
            expectation_model_log_fragment_probs += torch.log(
                # (F x S) sparse matrix @ (S x N) dense
                torch_sparse.spmm(
                    self.model.get_fragment_frequencies().indices(),
                    self.model.get_fragment_frequencies().values(),
                    self.model.get_fragment_frequencies().size()[0],
                    self.model.get_fragment_frequencies().size()[1],
                    softmax_x_t.t()
                ).t()
                # softmax_x_t @ self.model.get_fragment_frequencies().t()  # (N x S) @ (S x F) = (N x F)
            ).mv(phi_sum)
            # vi_log_expectation(
            #     self.fragment_posterior.phi[t_idx].indices(),
            #     self.fragment_posterior.phi[t_idx].values(),
            #     x_samples[t_idx, :, :],
            #     self.model.get_fragment_frequencies().indices(),
            #     self.model.get_fragment_frequencies().values(),
            #     self.model.get_fragment_space().size()
            # )

        elbo_samples = (model_gaussian_log_likelihoods
                        + expectation_model_log_fragment_probs
                        - posterior_gaussian_log_likelihoods)
        return elbo_samples.mean()

    def update_phi(self, x_samples: torch.Tensor):
        if cfg.model_cfg.use_sparse:
            return self.update_phi_sparse(x_samples)
        else:
            return self.update_phi_dense(x_samples)

    def update_phi_dense(self, x_samples: torch.Tensor):
        """
        This step represents the explicit solution of maximizing the ELBO of Q_phi (the mean-field portion of
        the read-to-fragment posteriors), given a particular solution of (samples from) Q_X.
        :param x_samples:
        :return:
        """
        W = self.model.get_fragment_frequencies()
        self.fragment_posterior.phi = []

        for t in range(self.model.num_times()):
            # phi_t is a (F x R) matrix, row-scaling. (e.g. multiply each (Fx1) column entrywise by length-F vector.)
            phi_t = self.data_likelihoods.matrices[t] * torch.exp(
                torch.mean(
                    torch.log(W @ softmax(x_samples[t], dim=1).transpose(0, 1)),
                    dim=1
                )
            ).unsqueeze(1)
            self.fragment_posterior.phi.append(normalize(phi_t, dim=0))

    def update_phi_sparse(self, x_samples: torch.Tensor):
        """
        Same as update_phi_dense, but accounts for the fact that the W and read_likelihoods[t] matrices are sparse.
        :param x_samples:
        :return:
        """
        W = self.model.get_fragment_frequencies()
        self.fragment_posterior.phi = []

        for t in range(self.model.num_times()):
            phi_t = self.data_likelihoods.matrices[t].scale_row(
                torch.exp(torch.mean(
                    torch.log(W @ softmax(x_samples[t], dim=1).t()),
                    dim=1
                )),
                dim=0
            )
            self.fragment_posterior.phi.append(normalize_sparse_2d(
                phi_t.indices,
                phi_t.values,
                phi_t.rows,
                phi_t.columns,
                0
            ).coalesce())

    def solve(self,
              optim_class=torch.optim.Adam,
              optim_args=None,
              iters=4000,
              num_samples=8000,
              print_debug_every=1,
              thresh_elbo=0.0,
              callbacks: Optional[List[Callable]] = None):
        if optim_args is None:
            optim_args = {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.}

        optimizer = optim_class(
            self.gaussian_posterior.trainable_parameters,
            **optim_args
        )

        logger.debug("BBVI algorithm started. (Correlation={corr}, Gradient method, Target iterations={it}, lr={lr}, n_samples={n_samples})".format(
            corr=self.correlation_type,
            it=iters,
            lr=optim_args["lr"],
            n_samples=num_samples
        ))

        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        last_elbo = float("-inf")
        elbo_diff = float("inf")
        k = 0
        while k < iters:
            k += 1
            time_est.stopwatch_click()

            x_samples, gaussian_log_likelihoods = self.gaussian_posterior.reparametrized_sample(
                num_samples=num_samples,
                output_log_likelihoods=True,
                detach_grad=False
            )  # (T x N x S)

            optimizer.zero_grad()
            with torch.no_grad():
                self.update_phi(x_samples.detach())

            elbo = self.elbo_marginal_gaussian(x_samples, gaussian_log_likelihoods)
            elbo_loss = -elbo  # Quantity to minimize. (want to maximize ELBO)
            elbo_loss.backward()
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback(k, x_samples, elbo.detach())

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            if k % print_debug_every == 0:
                logger.info("Iteration {iter} "
                            "| time left: {t:.2f} min. "
                            "| Last ELBO = {elbo:.2f}"
                            .format(iter=k,
                                    t=time_est.time_left() / 60000,
                                    elbo=elbo.item())
                            )

            elbo_value = elbo.detach().item()
            elbo_diff = elbo_value - last_elbo
            if abs(elbo_diff) < thresh_elbo * abs(last_elbo):
                logger.info("Convergence criteria |ELBO_diff| < {} * |last_ELBO| met; terminating early.".format(thresh_elbo))
                break
            last_elbo = elbo_value
        logger.info("Finished {k} iterations. | ELBO diff = {diff}".format(
            k=k,
            diff=elbo_diff
        ))





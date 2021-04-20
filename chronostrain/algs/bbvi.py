"""
  bbvi.py (pytorch implementation)
  Black-box Variational Inference
  Author: Younhun Kim

  This is an implementation of BBVI for the posterior q(X_1) q(X_2 | X_1) ...
  (Note: doesn't work as well as BBVI for mean-field assumption.)
"""
from typing import List, Iterable, Tuple, Union

import geotorch
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import softmax, softplus

from chronostrain.config import cfg
from chronostrain.util.data_cache import CacheTag
from chronostrain.algs.vi import AbstractPosterior
from chronostrain.model import GenerativeModel, Fragment
from chronostrain.model.io import TimeSeriesReads
from chronostrain.algs.base import AbstractModelSolver
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.util.math import normalize
from . import logger


class PositiveScaleLayer(torch.nn.Module):

    def __init__(self, size: torch.Size):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(size))

    def forward(self, input: torch.Tensor):
        return softplus(self.scale) * input


class GaussianPosterior(AbstractPosterior):
    def __init__(self, model: GenerativeModel):
        """
        Mean-field assumption:
        1) Parametrize X_1, ..., X_T as a Gaussian Process, with covariance kernel \Sigma_{t, t+1}.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param model: The generative model to use.
        """
        # Check: might need this to be a matrix, not a vector.
        self.model = model

        # ================= Parameters to optimize using gradients
        # self.means = [
        #     torch.nn.Parameter(
        #         torch.zeros(self.model.num_strains(), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype),
        #         requires_grad=True
        #     )
        #     for _ in range(self.model.num_times())
        # ]
        #
        # self.stdevs_sources = [
        #     torch.nn.Parameter(
        #         torch.zeros(self.model.num_strains(), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype),
        #         requires_grad=True
        #     )
        #     for _ in range(self.model.num_times())
        # ]
        # self.trainable_parameters = self.means + self.stdevs_sources

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks: List[torch.nn.Module] = []

        for _ in range(self.model.num_strains()):
            scaling_layer = PositiveScaleLayer(
                size=torch.Size(
                    (self.model.num_times(),)
                )
            )
            linear_layer = torch.nn.Linear(
                in_features=self.model.num_times(),
                out_features=self.model.num_times(),
                bias=True,
            )
            geotorch.orthogonal(linear_layer, "weight")
            self.reparam_networks.append(
                torch.nn.Sequential(
                    scaling_layer,
                    linear_layer
                )
            )

        self.trainable_parameters = []
        for network in self.reparam_networks:
            self.trainable_parameters += network.parameters()

        self.std_gaussian = MultivariateNormal(
            loc=torch.zeros(self.model.num_times(), dtype=torch.double, device=cfg.torch_cfg.device),
            covariance_matrix=torch.eye(
                self.model.num_times(),
                self.model.num_times(),
                dtype=cfg.torch_cfg.default_dtype,
                device=cfg.torch_cfg.device)
        )

    def sample(self, num_samples=1) -> torch.Tensor:
        return self._private_sample(
            num_samples=num_samples, output_log_likelihoods=False, detach_grad=True
        ).transpose(0,2)

    def _private_sample(self,
                        num_samples=1,
                        output_log_likelihoods=False,
                        detach_grad=False
                        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = [
            self.std_gaussian.sample(sample_shape=torch.Size([num_samples]))
            for _ in range(self.model.num_strains())
        ]

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[s].forward(std_gaussian_samples[s])
            for s in range(self.model.num_strains())
        ])  # (S x N x T)

        # samples = torch.stack([
        #     self.means[t].expand(num_samples, -1) + softplus(self.stdevs_sources[t]).expand(num_samples, -1) * std_gaussian_samples[t]
        #     for t in range(self.model.num_times())
        # ])

        if detach_grad:
            samples = samples.detach()

        if output_log_likelihoods:
            log_likelihoods = torch.stack([
                self.std_gaussian.log_prob(std_gaussian_samples[s])
                for s in range(self.model.num_strains())
            ])
            return samples, log_likelihoods
        else:
            return samples


class FragmentPosterior(object):
    def __init__(self, model: GenerativeModel):
        self.model = model

        # length-T list of (F x N_t) tensors.
        self.phi: List[torch.Tensor] = []

    def top_fragments(self, time_idx, read_idx, top=5) -> Iterable[Tuple[Fragment, float]]:
        topk_result = torch.topk(
            input=self.phi[time_idx][:, read_idx],
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
                 cache_tag: CacheTag):
        super().__init__(model, data, cache_tag)
        self.gaussian_posterior = GaussianPosterior(model=model)
        self.fragment_posterior = FragmentPosterior(model=model)  # time-indexed list of F x N tensors.

        for t_idx, read_likelihood_matrix in enumerate(self.read_likelihoods):
            sums = read_likelihood_matrix.sum(dim=0)

            zero_indices = {i.item() for i in torch.where(sums == 0)[0]}
            if len(zero_indices) > 0:
                logger.warn("[t = {}] Discarding reads with overall likelihood zero: {}".format(
                    self.model.times[t_idx],
                    ",".join([str(read_idx) for read_idx in zero_indices])
                ))

                leftover_indices = [
                    i
                    for i in range(len(data[t_idx]))
                    if i not in zero_indices
                ]
                self.read_likelihoods_tensors[t_idx] = read_likelihood_matrix[:, leftover_indices]

    def elbo_marginal_gaussian(self, x_samples: torch.Tensor, posterior_gaussian_log_likelihoods: torch.Tensor) -> torch.Tensor:
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

        # ======== log P(Xi)
        model_gaussian_log_likelihoods = self.model.log_likelihood_x(X=x_samples)

        # ======== E_{F ~ Qf}(log P(F|Xi))
        n_samples = x_samples.size()[1]
        expectation_model_log_fragment_probs = torch.zeros(size=(n_samples,), dtype=cfg.torch_cfg.default_dtype)
        for t_idx in range(self.model.num_times()):
            model_frag_likelihoods_t = softmax(
                x_samples[t_idx, :, :],  # (N x S)
                dim=1
            ).mm(
                self.model.get_fragment_frequencies().t()  # (S x F)
            ).log()  # (N x F)

            '''
            Expectation is \sum_{read} \sum_{frag} phi[read,frag] * logP(frag|X).
            For a speedup, we switch order of summation: \sum_{frag} logP(frag|X) * (\sum_{read} phi[read,frag])
            '''
            expectation_model_log_fragment_probs += model_frag_likelihoods_t.mv(
                self.fragment_posterior.phi[t_idx].sum(dim=1)  # length F
            )  # length N
        elbo_samples = (model_gaussian_log_likelihoods
                        + expectation_model_log_fragment_probs
                        - posterior_gaussian_log_likelihoods)
        return elbo_samples.mean()

    def update_phi(self, x_samples: torch.Tensor, smoothing=0.0):
        """
        This step represents the explicit solution of maximizing the ELBO of Q_phi (the mean-field portion of
        the read-to-fragment posteriors), given a particular solution of (samples from) Q_X.
        :param x_samples:
        :return:
        """
        W = self.model.get_fragment_frequencies()
        self.fragment_posterior.phi = []

        for t in range(self.model.num_times()):
            phi_t = self.read_likelihoods[t] * torch.exp(
                torch.mean(
                    torch.matmul(W, softmax(x_samples[t], dim=1).transpose(0, 1)),
                    dim=1
                )
            ).unsqueeze(1)
            self.fragment_posterior.phi.append(normalize(phi_t + smoothing, dim=0))

    def solve(self,
              optim_class=torch.optim.Adam,
              optim_args=None,
              iters=4000,
              num_samples=8000,
              print_debug_every=1,
              thresh_elbo=0.0,
              store_elbos: bool = False):

        if optim_args is None:
            optim_args = {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.}

        elbo_history = []

        optimizer = optim_class(
            self.gaussian_posterior.trainable_parameters,
            **optim_args
        )

        logger.debug("BBVI algorithm started. (Gradient method, Target iterations={it}, lr={lr}, n_samples={n_samples})".format(
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

            x_samples, gaussian_log_likelihoods = self.gaussian_posterior._private_sample(
                num_samples=num_samples,
                output_log_likelihoods=True,
                detach_grad=False
            )

            x_samples = x_samples.transpose(0, 2)

            optimizer.zero_grad()
            with torch.no_grad():
                self.update_phi(x_samples.detach())

            elbo = self.elbo_marginal_gaussian(x_samples, gaussian_log_likelihoods)
            elbo_loss = -elbo  # Quantity to minimize. (want to maximize ELBO)
            elbo_loss.backward()
            optimizer.step()

            if store_elbos:
                elbo_history.append(elbo.item())

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

        return elbo_history

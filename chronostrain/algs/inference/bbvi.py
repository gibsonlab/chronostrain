"""
  bbvi.py (pytorch implementation)
  Black-box Variational Inference
  Author: Younhun Kim

  This is an implementation of BBVI for the posterior q(X_1) q(X_2 | X_1) ...
  (Note: doesn't work as well as BBVI for mean-field assumption.)
"""
from typing import Iterator, Tuple, Optional, Callable, Dict, Union, List

from torch import Tensor
from torch.nn import functional
from torch.distributions import Normal

from chronostrain.config import cfg
from .base import AbstractModelSolver
from .vi import AbstractPosterior
from chronostrain.model import GenerativeModel, Fragment
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.util.math import *
from chronostrain.util.sparse import SparseMatrix, ColumnSectionedSparseMatrix, RowSectionedSparseMatrix
from chronostrain.database import StrainDatabase

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


# ============== Posteriors

class LinearTrilGaussian(torch.nn.Linear):
    def __init__(self, n_features: int, bias: bool, device=None, dtype=None):
        super().__init__(
            in_features=n_features,
            out_features=n_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.n_features = n_features

    @property
    def cholesky_part(self) -> torch.Tensor:
        x = torch.tril(self.weight, diagonal=-1)
        x[range(self.n_features), range(self.n_features)] = torch.exp(torch.diag(self.weight))
        # x[range(self.n_features), range(self.n_features)] = torch.diag(self.weight)
        return x

    def forward(self, input: Tensor) -> Tensor:
        return functional.linear(input, self.cholesky_part, self.bias)


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
        self.reparam_network = LinearTrilGaussian(
            n_features=self.model.num_times() * self.model.num_strains(),
            bias=True,
            device=cfg.torch_cfg.device
        )
        torch.nn.init.eye_(self.reparam_network.weight)
        self.trainable_parameters = self.reparam_network.parameters()
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def sample(self, num_samples=1) -> torch.Tensor:
        return self.reparametrized_sample(
            num_samples=num_samples, output_log_likelihoods=False, detach_grad=True
        )

    def mean(self) -> torch.Tensor:
        return self.reparam_network.bias.detach()

    def reparametrized_sample(self,
                              num_samples=1,
                              output_log_likelihoods=False,
                              detach_grad=False
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(num_samples, self.model.num_times() * self.model.num_strains())
        )

        """
        Reparametrization: x = a + bz, z ~ N(0,1).
        When computing log-likelihood p(x; a,b), it is important to keep a,b differentiable. e.g.
        output logp = f(a+bz; a, b) where f is the gaussian density N(a,b).
        """
        reparametrized_samples = self.reparam_network(std_gaussian_samples)

        if detach_grad:
            reparametrized_samples = reparametrized_samples.detach()

        if output_log_likelihoods:
            log_likelihoods = self.reparametrized_sample_log_likelihoods(reparametrized_samples)
            return reparametrized_samples.view(
                num_samples, self.model.num_times(), self.model.num_strains()
            ).transpose(0, 1), log_likelihoods
        else:
            return reparametrized_samples.view(
                num_samples, self.model.num_times(), self.model.num_strains()
            ).transpose(0, 1)

    def reparametrized_sample_log_likelihoods(self, samples):
        w = self.reparam_network.cholesky_part
        return torch.distributions.MultivariateNormal(
            loc=self.reparam_network.bias,
            covariance_matrix=w.mm(w.t())
        ).log_prob(samples)

    def log_likelihood(self, x: torch.Tensor) -> float:
        if len(x.size()) == 2:
            r, c = x.size()
            x = x.view(r, 1, c)
        return self.reparametrized_sample_log_likelihoods(x).detach()


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
            linear_layer = LinearTrilGaussian(
                n_features=self.model.num_strains(),
                bias=True,
                device=cfg.torch_cfg.device
            )
            torch.nn.init.eye_(linear_layer.weight)
            self.reparam_networks.append(linear_layer)

        self.trainable_parameters = []
        for network in self.reparam_networks:
            self.trainable_parameters += network.parameters()

        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def sample(self, num_samples=1) -> torch.Tensor:
        return self.reparametrized_sample(
            num_samples=num_samples, output_log_likelihoods=False, detach_grad=True
        )

    def mean(self) -> torch.Tensor:
        return torch.stack([
            self.reparam_networks[t].bias.detach()
            for t in range(self.model.num_times())
        ], dim=0)

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
            return samples, self.reparametrized_sample_log_likelihoods(samples)
        else:
            return samples

    def reparametrized_sample_log_likelihoods(self, samples):
        # samples is (T x N x S)
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.torch_cfg.device)
        for t in range(self.model.num_times()):
            samples_t = samples[t]
            linear = self.reparam_networks[t]
            try:
                w = linear.cholesky_part
                log_likelihood_t = torch.distributions.MultivariateNormal(
                    loc=linear.bias,
                    covariance_matrix=w.mm(w.t())
                ).log_prob(samples_t)
            except ValueError as e:
                w = linear.cholesky_part
                logger.error("Resulting covariance cholesky: (t={}) {}".format(t, w))
                raise e
            ans = ans + log_likelihood_t
        return ans

    def log_likelihood(self, x: torch.Tensor) -> float:
        if len(x.size()) == 2:
            r, c = x.size()
            x = x.view(r, 1, c)
        return self.reparametrized_sample_log_likelihoods(x).detach()


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
            linear_layer = LinearTrilGaussian(
                n_features=self.model.num_times(),
                bias=True,
                device=cfg.torch_cfg.device
            )
            torch.nn.init.eye_(linear_layer.weight)
            self.reparam_networks[s_idx] = linear_layer
        self.trainable_parameters = []
        for network in self.reparam_networks.values():
            self.trainable_parameters += network.parameters()

        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def sample(self, num_samples=1) -> torch.Tensor:
        return self.reparametrized_sample(
            num_samples=num_samples, output_log_likelihoods=False, detach_grad=True
        )

    def mean(self) -> torch.Tensor:
        return torch.stack([
            self.reparam_networks[s].bias.detach()
            for s in range(self.model.num_strains())
        ], dim=1)

    def reparametrized_sample(self,
                              num_samples=1,
                              output_log_likelihoods=False,
                              detach_grad=False
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.model.num_strains(), num_samples, self.model.num_times()),
        )

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[s_idx].forward(std_gaussian_samples[s_idx, :, :])
            for s_idx in range(self.model.num_strains())
        ], dim=0)

        if detach_grad:
            samples = samples.detach()

        if output_log_likelihoods:
            return samples.transpose(0, 2), self.reparametrized_sample_log_likelihoods(samples)
        else:
            return samples.transpose(0, 2)

    def reparametrized_sample_log_likelihoods(self, samples):
        # For this posterior, samples is (S x N x T).
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.torch_cfg.device)
        for s in range(self.model.num_strains()):
            samples_s = samples[s]
            linear = self.reparam_networks[s]
            w = linear.cholesky_part
            log_likelihood_s = torch.distributions.MultivariateNormal(
                loc=linear.bias,
                covariance_matrix=w.mm(w.t())
            ).log_prob(samples_s)
            ans = ans + log_likelihood_s
        return ans

    def log_likelihood(self, x: torch.Tensor) -> float:
        if len(x.size()) == 2:
            r, c = x.size()
            x = x.view(r, 1, c)
        return self.reparametrized_sample_log_likelihoods(x).detach()


class FragmentPosterior(object):
    def __init__(self, model: GenerativeModel, frag_index_map: Optional[Callable] = None):
        self.model = model

        # length-T list of (F x N_t) tensors.
        self.phi: List[Union[torch.Tensor, SparseMatrix]] = []
        self.frag_index_map = frag_index_map  # A mapping from internal frag indices to FragmentSpace indexing.

    def top_fragments(self, time_idx, read_idx, top=5) -> Iterator[Tuple[Fragment, float]]:
        phi_t = self.phi[time_idx]
        if isinstance(phi_t, SparseMatrix):
            # Assumes that this is a 1-d tensor (representing sparse values)
            read_slice = phi_t.sparse_slice(dim=1, idx=read_idx)

            sparse_topk = torch.topk(
                input=read_slice.values,
                k=min(top, read_slice.values.size()[0]),
                sorted=True
            )
            for sparse_idx, frag_prob in zip(sparse_topk.indices, sparse_topk.values):
                internal_frag_idx = read_slice.indices[0, sparse_idx]
                frag_idx = self.frag_index_map(internal_frag_idx, time_idx)
                yield self.model.fragments.get_fragment_by_index(frag_idx), frag_prob.item()
        elif isinstance(phi_t, torch.Tensor):
            topk_result = torch.topk(
                input=phi_t[:, read_idx],
                k=top,
                sorted=True
            )
            for frag_idx, frag_prob in zip(topk_result.indices, topk_result.values):
                yield self.model.fragments.get_fragment_by_index(frag_idx), frag_prob.item()
        else:
            raise RuntimeError("Unexpected type for fragment posterior parametrization `phi_t`.")


class BBVISolver(AbstractModelSolver):
    """
    An abstraction of a black-box VI implementation.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 num_cores: int = 1,
                 correlation_type: str = "time"):
        super().__init__(model, data, db, num_cores=num_cores)
        self.correlation_type = correlation_type
        if correlation_type == "time":
            self.gaussian_posterior = GaussianPosteriorTimeCorrelation(model=model)
        elif correlation_type == "strain":
            self.gaussian_posterior = GaussianPosteriorStrainCorrelation(model=model)
        elif correlation_type == "full":
            self.gaussian_posterior = GaussianPosteriorFullCorrelation(model=model)
        # elif correlation_type == "block-diagonal":
        #     self.gaussian_posterior = GaussianPosteriorBlockDiagonalCorrelation(model=model)
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        # self.model.get_fragment_frequencies()
        self._sparse_frag_freqs: List[RowSectionedSparseMatrix] = []  # time-indexed list of (F' x S) matrices.
        if cfg.model_cfg.use_sparse:
            frag_freqs: SparseMatrix = self.model.fragment_frequencies_sparse  # F x S
            for t_idx in range(model.num_times()):
                # Sparsity transformation (R^F -> R^{Support}), matrix size = (F' x F)
                projector = self.data_likelihoods.projectors[t_idx]

                self._sparse_frag_freqs.append(
                    RowSectionedSparseMatrix.from_sparse_matrix(projector.sparse_mul(frag_freqs))
                )

            self.fragment_posterior = FragmentPosterior(
                model=model,
                frag_index_map=lambda fidx, tidx: self.data_likelihoods.supported_frags[tidx][fidx].item()
            )  # time-indexed list of F x N tensors.
        else:
            self.fragment_posterior = FragmentPosterior(model=model)  # time-indexed list of F x N tensors.

    def elbo_marginal_gaussian(self,
                               x_samples: torch.Tensor,
                               posterior_gaussian_log_likelihoods: torch.Tensor,
                               eps_smoothing: float) -> torch.Tensor:
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
        :param eps_smoothing: A small epsilon parameter to smooth out estimates (e.g. normalize eps + phi,
            instead of phi which may contain columns of all zeros due to numerical precision.)
        :return: An estimate of the ELBO, using the provided samples via the above formula.
        """
        if cfg.model_cfg.use_sparse:
            return self.elbo_marginal_gaussian_sparse(x_samples, posterior_gaussian_log_likelihoods, eps_smoothing)
        else:
            return self.elbo_marginal_gaussian_dense(x_samples, posterior_gaussian_log_likelihoods, eps_smoothing)

    def elbo_marginal_gaussian_dense(self,
                                     x_samples: torch.Tensor,
                                     posterior_gaussian_log_likelihoods: torch.Tensor,
                                     eps_smoothing: float) -> torch.Tensor:
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

            model_frag_likelihoods_t = log_mm_exp(
                torch.log(softmax(x_samples[t_idx], dim=1)),  # (N x S)
                self.model.fragment_frequencies_dense.t()  # (S x F)
            )  # (N x F)

            expectation_model_log_fragment_probs += model_frag_likelihoods_t.mv(
                normalize(self.fragment_posterior.phi[t_idx] + eps_smoothing, dim=0).sum(dim=1)  # length F
            )  # length N

        elbo_samples = (model_gaussian_log_likelihoods
                        + expectation_model_log_fragment_probs
                        - posterior_gaussian_log_likelihoods)
        return elbo_samples.mean()

    def elbo_marginal_gaussian_sparse(self,
                                      x_samples: torch.Tensor,
                                      posterior_gaussian_log_likelihoods: torch.Tensor,
                                      eps_smoothing: float
                                      ) -> torch.Tensor:
        """
        Same as dense implementation, but computes the middle term E_{F ~ Qf}(log P(F|Xi)) using monte-carlo samples
        of F from phi.
        """
        elbo_sum = posterior_gaussian_log_likelihoods.sum()

        # ======== log P(Xi)
        model_gaussian_log_likelihoods = self.model.log_likelihood_x(X=x_samples).sum()
        elbo_sum += model_gaussian_log_likelihoods.sum()

        # ======== E_{F ~ Qf}(log P(F|Xi))
        n_samples = x_samples.size()[1]
        for t_idx in range(self.model.num_times()):
            # softmax_x_t = torch.softmax(x_samples[t_idx, :, :], dim=1)  # (N x S)

            phi_t: RowSectionedSparseMatrix = self.fragment_posterior.phi[t_idx]
            elbo_sum = total_expectation_frag_ll(
                phi_t,
                self._sparse_frag_freqs[t_idx],
                torch.log(torch.softmax(x_samples[t_idx, :, :], dim=1)),
                cumulative_answer=elbo_sum
            )  # This function automatically returns elbo_sum + (partial answer), due to the limitation with jit.
            # print(elbo_sum)

            # softmax_x_t = torch.softmax(x_samples[t_idx, :, :], dim=1)  # (N x S)
            # phi_sum = normalize(self.fragment_posterior.phi[t_idx].to_dense() + eps_smoothing, dim=0).sum(dim=1)  # (length F')
            #
            # # This computes E_phi[ log(F|X) ].
            # expectation_model_log_fragment_probs += log_spmm_exp(
            #     self._sparse_frag_freqs[t_idx],  # (F x S)
            #     torch.log(softmax_x_t).t()  # (S x N)
            # ).t().mv(phi_sum)  # (N x F) @ (F)

        # elbo_sum = (model_gaussian_log_likelihoods
        #             + expectation_model_log_fragment_probs
        #             - posterior_gaussian_log_likelihoods.sum())
        return elbo_sum * (1 / n_samples)

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
        W = self.model.fragment_frequencies_dense
        self.fragment_posterior.phi = []

        for t in range(self.model.num_times()):
            phi_t = self.data_likelihoods.matrices[t].exp() * torch.exp(
                torch.mean(
                    log_mm_exp(W, torch.log(softmax(x_samples[t], dim=1).t())),
                    dim=1
                )
            ).unsqueeze(1)
            self.fragment_posterior.phi.append(phi_t)

    def update_phi_sparse(self, x_samples: torch.Tensor):
        """
        Same as update_phi_dense, but accounts for the fact that the W and read_likelihoods[t] matrices are sparse.
        :param x_samples:
        :return:
        """
        self.fragment_posterior.phi = []

        for t in range(self.model.num_times()):
            phi_t = construct_phi(
                self._sparse_frag_freqs[t],
                self.data_likelihoods.matrices[t],
                torch.log(softmax(x_samples[t], dim=1))
            )
            self.fragment_posterior.phi.append(phi_t)

            # print(phi_t.to_dense())

            # phi_t: SparseMatrix = self.data_likelihoods.matrices[t].exp().scale_row(
            #     torch.exp(
            #         torch.mean(
            #             log_spmm_exp(
            #                 self._sparse_frag_freqs[t],  # (F' x S)
            #                 torch.log(softmax(x_samples[t], dim=1)).t()  # (S x N)
            #             ),  # Result is #(F' x N)
            #             dim=1
            #         )
            #     ),
            #     dim=0
            # )
            # self.fragment_posterior.phi.append(phi_t)




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

        logger.debug(
            "BBVI algorithm started. "
            "(Correlation={corr}, Gradient method, Target iterations={it}, lr={lr}, n_samples={n_samples})".format(
                corr=self.correlation_type,
                it=iters,
                lr=optim_args["lr"],
                n_samples=num_samples
            )
        )

        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        last_elbo = float("-inf")
        elbo_diff = float("inf")
        k = 0

        while k < iters:
            k += 1
            time_est.stopwatch_click()

            _t = RuntimeEstimator(total_iters=iters, horizon=1)
            _t.stopwatch_click()
            x_samples, gaussian_log_likelihoods = self.gaussian_posterior.reparametrized_sample(
                num_samples=num_samples,
                output_log_likelihoods=True,
                detach_grad=False
            )  # (T x N x S)

            optimizer.zero_grad()
            with torch.no_grad():
                self.update_phi(x_samples.detach())

            elbo = self.elbo_marginal_gaussian(x_samples, gaussian_log_likelihoods, eps_smoothing=1e-30)

            elbo_loss = -elbo  # Quantity to minimize. (want to maximize ELBO)
            elbo_loss.backward()
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback(k, x_samples, elbo.detach())

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            if k % print_debug_every == 0:
                logger.info(
                    "Iteration {iter} | time left: {t:.2f} min. | Last ELBO = {elbo:.2f}".format(
                        iter=k,
                        t=time_est.time_left() / 60000,
                        elbo=elbo
                    )
                )

            elbo_value = elbo.detach()
            elbo_diff = elbo_value - last_elbo
            if abs(elbo_diff) < thresh_elbo * abs(last_elbo):
                logger.info("Convergence criteria |ELBO_diff| < {} * |last_ELBO| met; terminating early.".format(
                    thresh_elbo
                ))
                break
            last_elbo = elbo_value
        logger.info("Finished {k} iterations. | ELBO diff = {diff}".format(
            k=k,
            diff=elbo_diff
        ))

        if cfg.torch_cfg.device == torch.device("cuda"):
            logger.info(
                "BBVI CUDA memory -- [MaxAlloc: {} MiB]".format(
                    torch.cuda.max_memory_allocated(cfg.torch_cfg.device) / 1048576
                )
            )
        else:
            logger.debug(
                "BBVI CPU memory usage -- [Not implemented]"
            )


# ======================= Torch JIT helpers
@torch.jit.script
def ll_conditional_frag_given_xt(
        n_strains: int,
        ff_indices: torch.Tensor,
        ff_values: torch.Tensor,
        frag_locs: torch.Tensor,
        log_x_t: torch.Tensor,
        empty_ll_values: float
) -> torch.Tensor:
    """
    Computes [log P(F=f | X_t)], for a particular choice of f (specified by frag_locs from a RowSectionedSparseMatrix).
    """
    supported_strains = ff_indices[1, frag_locs]

    ff_buffer = torch.full(
        [n_strains],
        empty_ll_values,
        dtype=torch.float,
        device=ff_values.device
    )
    ff_buffer[supported_strains] = ff_values[frag_locs]

    return torch.logsumexp(
        log_x_t + ff_buffer.unsqueeze(0),
        dim=1
    )  # (length N)


def total_expectation_frag_ll(phi_t: RowSectionedSparseMatrix,
                              frag_freqs: RowSectionedSparseMatrix,
                              log_x_t: torch.Tensor,
                              cumulative_answer: torch.Tensor,
                              empty_ll_values: float = -1e7) -> torch.Tensor:
    """
    :param phi_t: An (F x R) sparse matrix, representing the approximate (log-) posterior.
    :param frag_freqs: An (F x S) sparse matrix.
    :param log_x_t: An (N x S) dense tensor.
    :return: A length-N tensor representing a vector of E_(f ~ phi_t)[ log P(F=f | X_i^t) ], for each i-th sample X_i.
    """
    return total_expectation_frag_ll_helper(
        phi_t.rows,
        phi_t.values,
        phi_t.locs_per_row,
        frag_freqs.indices,
        frag_freqs.values,
        frag_freqs.locs_per_row,
        log_x_t,
        cumulative_answer,
        empty_ll_values
    )


@torch.jit.script
def total_expectation_frag_ll_helper(
        n_fragments: int,
        phi_values: torch.Tensor,
        phi_locs_per_row: List[torch.Tensor],
        ff_indices: torch.Tensor,
        ff_values: torch.Tensor,
        ff_locs_per_row: List[torch.Tensor],
        log_x_t: torch.Tensor,
        cumulative_answer: torch.Tensor,
        empty_ll_values: float
) -> torch.Tensor:
    n_strains = log_x_t.size(1)

    for f in torch.arange(0, n_fragments, 1):
        """
        Let X_n(t) represent the n-th sample.
        This part represents the calculation (for a particular timepoint t)
            Σ_{n} Σ_{r} E_{f ~ φ(r)} [log P(F_r = f | X_n(t))]
            = Σ_{n} Σ_{r} Σ_{f} [φ_{f,r} * ll_factor(t, f, n)]
            = Σ_{f} Σ_{n} Σ_{r} [φ_{f,r} * ll_factor(t, f, n)]
            = Σ_{f} Σ_{n} ll_factor(t, f, n) * [Σ_{r} φ_{f,r}]
            = Σ_{f} {[Σ_{n} ll_factor(t, f, n)] * [Σ_{r} φ_{f,r}]}
            
        Recall that phi here is a LOG likelihood value.
        """
        ll_factor_total = ll_conditional_frag_given_xt(
            n_strains, ff_indices, ff_values, ff_locs_per_row[f], log_x_t, empty_ll_values
        ).sum()  # length N

        phi_total = phi_values[phi_locs_per_row[f]].exp().sum()  # length = R' = (# of supported reads of frag f)
        cumulative_answer += ll_factor_total * phi_total
    return cumulative_answer


@torch.jit.script
def construct_phi_jit_helper(
        n_fragments: int,
        n_reads: int,
        ff_indices: torch.Tensor,
        ff_values: torch.Tensor,
        ff_locs_per_row: List[torch.Tensor],
        data_ll_indices: torch.Tensor,
        data_ll_values: torch.Tensor,
        data_ll_locs_per_col: List[torch.Tensor],
        log_x_t: torch.Tensor,
        empty_ll_values: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Jit-accelerated implementation of the following idea:

    Let data_likelihoods_t be M.

    For each supported frag f:
        β[f] = E[P(F=f | X_t)] = mean(γ_1, ..., γ_N), where γ_n = logsumexp(W[f, :] + X[:, n])
        (note: β is pre-computed beforehand for each f.)

        For each read r:
        For each supported frag f of r (with respect to M):
            φ[f, r] = M[f, r] * β[f]
    """
    n_strains = log_x_t.size(1)
    device = log_x_t.device
    frag_expected_ll = torch.empty([n_fragments], dtype=torch.float, device=device)

    for f in torch.arange(0, n_fragments, 1):
        ll_factor = ll_conditional_frag_given_xt(
            n_strains, ff_indices, ff_values, ff_locs_per_row[f], log_x_t, empty_ll_values
        )
        frag_expected_ll[f] = ll_factor.mean()

    arr_indices = []
    arr_values = []
    for r in torch.arange(0, n_reads, 1):
        support_locs = data_ll_locs_per_col[r]
        target_frags = data_ll_indices[0][support_locs]

        target_values = frag_expected_ll[target_frags] + data_ll_values[support_locs]

        # represents log(ε_{r,f} * exp(E_X[log P(F=f | X)])) = log(ε_{r,f}) + E_X[log P(F=f | X)]
        # target_values = frag_expected_ll[support_locs] + data_ll_values[support_locs]
        arr_indices.append(
            torch.stack([
                target_frags,
                torch.full([target_frags.size(0)], r, device=device, dtype=target_frags.dtype)
            ])
        )
        arr_values.append(target_values)

    return (
        torch.concat(arr_indices, dim=1),
        torch.concat(arr_values)
    )


def construct_phi(frag_freqs: RowSectionedSparseMatrix,
                  data_ll_t: ColumnSectionedSparseMatrix,
                  log_x_t: torch.Tensor,
                  empty_ll_values: float = -1e7) -> RowSectionedSparseMatrix:
    """
    :param frag_freqs: An (F x S) row-sectioned sparse matrix.
    :param data_ll_t: An (F x R) column-sectioned sparse matrix.
    :param log_x_t: An (N x S) dense tensor.
    :param empty_ll_values: A default value to fill in for empty entries in the log-likelihood frag frequency matrix.
    :return: A row-sectioned (F x R) sparse matrix representing the (log-) fragment posterior phi_t
    """

    indices, values = construct_phi_jit_helper(
        data_ll_t.rows,
        data_ll_t.columns,
        frag_freqs.indices,
        frag_freqs.values,
        frag_freqs.locs_per_row,
        data_ll_t.indices,
        data_ll_t.values,
        data_ll_t.locs_per_column,
        log_x_t,
        empty_ll_values
    )

    # print("********************************")
    # print(values)
    # print(indices)
    # print("ROWS: ", data_ll_t.rows)
    # print("COLS: ", data_ll_t.columns)

    # Normalize each column.
    phi_unnormalized = ColumnSectionedSparseMatrix(
        indices=indices,
        values=values,
        dims=(data_ll_t.rows, data_ll_t.columns),
        force_coalesce=False
    )
    normalized_values = normalize_columns(
        x_values=phi_unnormalized.values,
        x_locs_per_col=phi_unnormalized.locs_per_column,
        n_columns=phi_unnormalized.columns
    )

    return RowSectionedSparseMatrix(
        indices=indices,
        values=normalized_values,
        dims=(data_ll_t.rows, data_ll_t.columns),
        force_coalesce=False
    )


@torch.jit.script
def normalize_columns(
        x_values: torch.Tensor,
        x_locs_per_col: List[torch.Tensor],
        n_columns: int
) -> torch.Tensor:
    """
    :return: The column-normalized version of x_values.
    """
    normalized_x_values = torch.empty(
        x_values.size(),
        device=x_values.device,
        dtype=x_values.dtype
    )
    for c in torch.arange(0, n_columns, 1):
        support_locs = x_locs_per_col[c]
        normalized_x_values[support_locs] = x_values[support_locs] - torch.logsumexp(x_values[support_locs], dim=0)
    return normalized_x_values

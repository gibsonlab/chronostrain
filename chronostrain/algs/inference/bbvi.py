"""
  bbvi.py (pytorch implementation)
  Black-box Variational Inference
  Author: Younhun Kim
"""
from typing import Iterator, Tuple, Optional, Callable, Union, List, Dict

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
from chronostrain.util.sparse import SparseMatrix, RowSectionedSparseMatrix
from chronostrain.database import StrainDatabase

from chronostrain.config.logging import create_logger
from ..subroutines.likelihoods import SparseDataLikelihoods
from ...util.sparse.sliceable import BBVIOptimizedSparseMatrix

logger = create_logger(__name__)


def log_softmax(x_samples: torch.Tensor, t: int) -> torch.Tensor:
    # x_samples: (T x N x S) tensor.
    return x_samples[t] - torch.logsumexp(x_samples[t], dim=1, keepdim=True)


class LogMMExpDenseSPModel(torch.nn.Module):
    """
    Represents a Module which represents
        f_A(X) = log_matmul_exp(X, A)
    where A is a (D x E) SparseMatrix, and X is an (N x D) matrix.
    """
    def __init__(self, sparse_right_matrix: SparseMatrix):
        super().__init__()
        self.A_indices: torch.Tensor = sparse_right_matrix.indices
        self.A_values: torch.Tensor = sparse_right_matrix.values
        self.A_rows: int = int(sparse_right_matrix.rows)
        self.A_columns: int = int(sparse_right_matrix.columns)

        self.nz_targets: List[torch.Tensor] = []
        self.target_cols: List[torch.Tensor] = []

        for target_idx in range(self.A_rows):
            nz_targets_k = torch.where(sparse_right_matrix.indices[0] == target_idx)[0]

            self.nz_targets.append(nz_targets_k)
            self.target_cols.append(self.A_indices[1, nz_targets_k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = torch.full(
            fill_value=-float('inf'),
            size=[x.shape[0], self.A_columns],
            device=self.A_values.device,
            dtype=self.A_values.dtype
        )

        for target_idx in range(self.A_rows):
            """
            Given a target index k, compute the k-th summand of the dot product <u,v> = SUM_k u_k v_k,
            for each row u of x, and each column v of y.

            Note that k here specifies a column of x, and a row of y.
            """
            nz_targets: torch.Tensor = self.nz_targets[target_idx]
            target_cols: torch.Tensor = self.target_cols[target_idx]

            next_sum: torch.Tensor = x[:, target_idx].view(-1, 1) + self.A_values[nz_targets].view(1, -1)
            result[:, target_cols] = torch.logsumexp(
                torch.stack([result[:, target_cols], next_sum], dim=0),
                dim=0
            )

        return result


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

    def forward(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.cholesky_part, self.bias)


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
        reparametrized_samples = self.reparam_network.forward(std_gaussian_samples)

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
        try:
            return torch.distributions.MultivariateNormal(
                loc=self.reparam_network.bias,
                scale_tril=w
            ).log_prob(samples)
        except ValueError:
            logger.error(f"Problem while computing log MV log-likelihood.")
            raise

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

        for _ in range(self.model.num_times()):
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
                    scale_tril=w
                ).log_prob(samples_t)
            except ValueError:
                logger.error(f"Problem while computing log MV log-likelihood of time index {t}.")
                raise
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
            torch.nn.init.constant_(linear_layer.bias, 0)
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
            try:
                log_likelihood_s = torch.distributions.MultivariateNormal(
                    loc=linear.bias,
                    scale_tril=w
                ).log_prob(samples_s)
            except ValueError:
                logger.error(f"Problem while computing log MV log-likelihood of strain index {s}.")
                raise
            ans = ans + log_likelihood_s
        return ans

    def log_likelihood(self, x: torch.Tensor) -> float:
        if len(x.size()) == 2:
            r, c = x.size()
            x = x.view(r, 1, c)
        return self.reparametrized_sample_log_likelihoods(x).detach()


class FragmentPosterior(object):
    def __init__(self,
                 model: GenerativeModel,
                 reads: TimeSeriesReads,
                 sparse_data_likelihoods: SparseDataLikelihoods,
                 frag_index_map: Optional[Callable] = None):
        self.model = model
        self.reads = reads

        # length-T list of (F x N_t) tensors.
        self.phi: List[BBVIOptimizedSparseMatrix] = [
            m.copy_pattern()
            for m in sparse_data_likelihoods.sparse_matrices()
        ]
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

    def renormalize(self, t_idx: int):
        phi_t = self.phi[t_idx]  # (F x R)

        # invoke torch-scatter to compute columnwise mins.
        scale_values = phi_t.columnwise_max()

        col_logsumexp = scale_values + SparseMatrix(
            indices=phi_t.indices,
            values=phi_t.values - scale_values[phi_t.indices[1]],
            dims=(phi_t.rows, phi_t.columns),
            force_coalesce=False
        ).exp().sum(dim=0).log()

        phi_t.update_values(phi_t.values - col_logsumexp[phi_t.indices[1]])

        #  TODO DEBUG
        # if torch.isnan(
        #         phi_t.exp().sum(dim=1)
        # ).sum() > 0:
        #     phi_t.values = old_values
        #     print("Found NANs in Phi, timepoint = {}".format(t_idx))
        #     print("scale_values: {}".format(scale_values))
        #     print("col_logsumexp: {}".format(col_logsumexp))
        #
        #     offending_col = torch.where(torch.isinf(col_logsumexp))[0][0]
        #     print(f"Offending column = {offending_col}")
        #     print(f"column {offending_col} entries:")
        #     for loc in torch.where(phi_t.indices[1] == offending_col)[0]:
        #         row = phi_t.indices[0, loc]
        #         col = offending_col
        #         val = phi_t.values[loc]
        #         print(f"\tPhi_{t_idx}[{row}, {col}] = {val}")
        #
        #     from pathlib import Path
        #     phi_t.save(Path("bad_phi_t.npz"))
        #
        #     exit(1)


class BBVISolver(AbstractModelSolver):
    """
    An abstraction of a black-box VI implementation.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 frag_chunk_size: int = 100,
                 num_cores: int = 1,
                 correlation_type: str = "time"):
        super().__init__(model, data, db, frag_chunk_size=frag_chunk_size, num_cores=num_cores)
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

        self.frag_chunk_size = frag_chunk_size
        self.log_mm_exp_models: List[List[LogMMExpDenseSPModel]] = [
            [] for _ in range(model.num_times())
        ]

        logger.debug("Initializing BBVI data structures.")
        if not cfg.model_cfg.use_sparse:
            raise NotImplementedError("BBVI only supports sparse data structures.")

        frag_freqs: SparseMatrix = self.model.fragment_frequencies_sparse  # F x S
        for t_idx in range(model.num_times()):
            # Sparsity transformation (R^F -> R^{Support}), matrix size = (F' x F)
            projector = self.data_likelihoods.projectors[t_idx]

            n_chunks = 0
            for sparse_chunk in BBVIOptimizedSparseMatrix.optimize_from_sparse_matrix(
                    projector.sparse_mul(frag_freqs),
                    row_chunk_size=self.frag_chunk_size
            ).chunks:
                n_chunks += 1
                self.log_mm_exp_models[t_idx].append(
                    LogMMExpDenseSPModel(sparse_chunk.t())
                )

            logger.debug(f"Divided {projector.rows} x {frag_freqs.columns} sparse matrix "
                         f"into {n_chunks} chunks.")

        self.fragment_posterior = FragmentPosterior(
            model=model,
            reads=data,
            sparse_data_likelihoods=self.data_likelihoods,
            frag_index_map=lambda fidx, tidx: self.data_likelihoods.supported_frags[tidx][fidx].item()
        )

    def elbo_marginal_gaussian(self,
                               x_samples: torch.Tensor,
                               posterior_gaussian_log_likelihoods: torch.Tensor
                               ) -> Iterator[torch.Tensor]:
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

        """
        ELBO original formula:
            E_Q[P(X)] - E_Q[Q(X)] + E_{F ~ phi} [log P(F | X)]
        
        To save memory on larger frag spaces, split the ELBO up into several pieces.
        """
        n_samples = x_samples.size()[1]
        # ======== -log Q(X), monte-carlo
        yield posterior_gaussian_log_likelihoods.sum() * (-1 / n_samples)

        # ======== log P(X)
        model_gaussian_log_likelihoods = self.model.log_likelihood_x(X=x_samples)
        yield model_gaussian_log_likelihoods.sum() * (1 / n_samples)

        # ======== E_{F ~ Qf}(log P(F|Xi))
        for t_idx in range(self.model.num_times()):
            # =========== NEW IMPLEMENTATION: chunks
            log_softmax_xt = log_softmax(x_samples, t=t_idx)

            for chunk_idx, phi_chunk in enumerate(
                    self.fragment_posterior.phi[t_idx].chunks
            ):
                e = self.elbo_chunk_helper(
                    t_idx,
                    chunk_idx,
                    log_softmax_xt,
                    phi_chunk
                ).sum() * (1 / n_samples)

                if torch.isinf(e) or torch.isnan(e):
                    logger.debug("About to throw exception.")

                    # These should contain no infs/nans.
                    logger.debug("Phi chunk sum: {}".format(
                        str(phi_chunk.exp().sum(dim=1))
                    ))

                    raise ValueError(f"Invalid ELBO value {e.item()} found for t_idx: {t_idx}, chunk_idx: {chunk_idx}")
                yield e

    def elbo_chunk_helper(self,
                          t_idx: int,
                          chunk_idx: int,
                          log_softmax_x_t: torch.Tensor,
                          phi_chunk: RowSectionedSparseMatrix) -> torch.Tensor:
        """
        This computes the following partial ELBO quantity:

            Σ_{n} Σ_{r} Σ_{f in CHUNK} [φ_{f,r} * Log <W_{f,.},X_n>]
            = Σ_{f in CHUNK} Σ_{n} Σ_{r} [φ_{f,r} * Log <W_{f,.},X_n>]
            = Σ_{f in CHUNK} [ Σ_{n} Log <W_{f,.},X_n> ] * [ Σ_{r} φ_{f,r} ]

        Assumes phi is already normalized, and that it represents the LOG posterior values.
        """
        return torch.dot(
            phi_chunk.exp().sum(dim=1),  # length (CHUNK_SZ)
            self.log_mm_exp_models[t_idx][chunk_idx].forward(log_softmax_x_t).sum(dim=0)
        )

    def update_phi(self, x_samples: torch.Tensor):
        """
        This step represents the explicit solution of maximizing the ELBO of Q_phi (the mean-field portion of
        the read-to-fragment posteriors), given a particular solution of (samples from) Q_X.
        :param x_samples:
        :return:
        """

        for t in range(self.model.num_times()):
            log_softmax_xt = log_softmax(x_samples, t=t)

            for chunk_idx, logmmexp_model in enumerate(self.log_mm_exp_models[t]):
                # The monte carlo approximation
                mc_expectation_ll = torch.mean(
                    logmmexp_model.forward(log_softmax_xt),
                    dim=0
                )  # Output: length CHUNK_SZ

                # for chunk_idx, data_ll_chunk in enumerate(self.data_likelihoods.matrices[t].chunks):
                data_ll_chunk = self.data_likelihoods.matrices[t].chunks[chunk_idx]
                phi_t_chunk = RowSectionedSparseMatrix(
                    indices=data_ll_chunk.indices,
                    values=data_ll_chunk.values + mc_expectation_ll[data_ll_chunk.indices[0]],
                    dims=(data_ll_chunk.rows, data_ll_chunk.columns),
                    force_coalesce=False,
                    _explicit_locs_per_row=data_ll_chunk.locs_per_row
                )

                self.fragment_posterior.phi[t].collect_chunk(chunk_idx, phi_t_chunk)

            self.fragment_posterior.renormalize(t)

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

            try:
                x_samples, gaussian_log_likelihoods = self.gaussian_posterior.reparametrized_sample(
                    num_samples=num_samples,
                    output_log_likelihoods=True,
                    detach_grad=False
                )  # (T x N x S)

                optimizer.zero_grad()
                with torch.no_grad():
                    self.update_phi(x_samples.detach())

                elbo_value = 0.0
                for elbo_chunk in self.elbo_marginal_gaussian(x_samples, gaussian_log_likelihoods):
                    elbo_loss_chunk = -elbo_chunk
                    elbo_loss_chunk.backward(retain_graph=True)
                    optimizer.step()

                    # Save float value for callbacks.
                    elbo_value += elbo_chunk.item()
            except ValueError:
                logger.error(f"Encountered ValueError while performing BBVI optimization at iteration {k}.")
                raise

            if callbacks is not None:
                for callback in callbacks:
                    callback(k, x_samples, elbo_value)

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            if k % print_debug_every == 0:
                logger.info(
                    "Iteration {iter} | time left: {t:.2f} min. | Last ELBO = {elbo:.2f}".format(
                        iter=k,
                        t=time_est.time_left() / 60000,
                        elbo=elbo_value
                    )
                )

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

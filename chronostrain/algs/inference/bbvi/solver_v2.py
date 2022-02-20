from typing import Optional, Callable, List, Iterator, Tuple

import torch

from chronostrain.database import StrainDatabase
from chronostrain.model import Fragment
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.algs.subroutines.likelihoods import SparseDataLikelihoods
from chronostrain.util.sparse import SparseMatrix
from chronostrain.util.sparse.sliceable import BBVIOptimizedSparseMatrix, RowSectionedSparseMatrix

from .base import AbstractBBVI
from .. import AbstractModelSolver
from .posteriors import *
from .util import log_softmax, LogMMExpDenseSPModel

from chronostrain.config import cfg, create_logger
logger = create_logger(__name__)


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


class BBVISolverV2(AbstractModelSolver, AbstractBBVI):
    """
    The BBVI implementation capturing the joint posterior p(X,F|R) via the
    mean-field family q(X)q(F).
    """
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 frag_chunk_size: int = 100,
                 num_cores: int = 1,
                 correlation_type: str = "time"):
        logger.info("Initializing V2 solver (Mean field posterior X,F)")
        AbstractModelSolver.__init__(
            self,
            model,
            data,
            db,
            frag_chunk_size=frag_chunk_size,
            num_cores=num_cores
        )

        self.correlation_type = correlation_type
        if correlation_type == "time":
            posterior = GaussianPosteriorTimeCorrelation(model=model)
        elif correlation_type == "strain":
            posterior = GaussianPosteriorStrainCorrelation(model=model)
        elif correlation_type == "full":
            posterior = GaussianPosteriorFullCorrelation(model=model)
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        AbstractBBVI.__init__(
            self,
            posterior,
            device=cfg.torch_cfg.device
        )

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
                # sparse_chunk: (CHUNK_SZ x S)
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

    def elbo(self,
             x_samples: torch.Tensor,
             posterior_sample_lls: torch.Tensor
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
        :param posterior_sample_lls: A length-N (one-dimensional) tensor of the joint log-likelihood
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
        yield posterior_sample_lls.sum() * (-1 / n_samples)

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

    def optimize_step(self,
                      samples: torch.Tensor,
                      posterior_sample_lls: torch.Tensor,
                      optimizer: torch.optim.Optimizer):
        with torch.no_grad():
            self.update_phi(samples.detach())
        return super().optimize_step(samples, posterior_sample_lls, optimizer)

    def solve(self,
              optimizer: torch.optim.Optimizer,
              lr_scheduler,
              num_epochs: int = 1,
              iters: int = 4000,
              num_samples: int = 8000,
              min_lr: float = 1e-4,
              callbacks: Optional[List[Callable[[int, torch.Tensor, float], None]]] = None):
        self.optimize(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iters=iters,
            num_epochs=num_epochs,
            num_samples=num_samples,
            min_lr=min_lr,
            callbacks=callbacks
        )

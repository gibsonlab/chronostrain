"""
vi.py
This is the second-order approximation solution for VI derived in a previous writeup.
(Note: doesn't work as well as mean-field BBVI.)
"""


from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
from torch.nn.functional import softmax
from torch.distributions import MultivariateNormal, Categorical

from chronostrain.config import cfg
from chronostrain.util.logger import logger
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.reads import SequenceRead
from chronostrain.algs.base import AbstractModelSolver
from chronostrain.util.benchmarking import RuntimeEstimator


class AbstractVariationalPosterior(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, num_samples: int = 1) -> List[torch.Tensor]:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed list of (N x S) abundance samples.
        """
        pass


# ========================== Implementations ===========================
class MeanFieldPosterior(AbstractVariationalPosterior):

    def __init__(self,
                 model: GenerativeModel,
                 read_counts: List[int],
                 num_update_samples: int,
                 read_likelihoods: List[torch.Tensor],
                 clipping: float = float("inf"),
                 stdev_scale: float = List[float]):
        self.model = model
        self.read_counts = read_counts
        self.num_update_samples = num_update_samples
        self.read_likelihoods = read_likelihoods
        self.clipping = clipping
        self.stdev_scale = stdev_scale

        # Variational parameters
        self.mu = torch.zeros(size=[model.num_strains()-1], device=cfg.torch_cfg.device)
        self.phi = [
            (1 / model.num_fragments()) * torch.ones(size=[model.num_fragments(), read_counts[t]], device=cfg.torch_cfg.device)
            for t in range(model.num_times())
        ]  # each is an F x R_t tensor.

    def deriv_precomputation(self, f: int, X: torch.Tensor, clipping=float("inf")):
        S = self.model.num_strains()

        W_f = self.model.get_fragment_frequencies()[f]  # S-dim vector.
        sigma = softmax(X, dim=1)  # N x S
        Wf_dot_sigma = sigma.mv(W_f)  # N-dim vector.

        sigma_deriv = self.sigmoid_derivative(X)  # N x S x S

        # Gradient clipping for stability.
        deriv_norm = sigma_deriv.norm(p=2, dim=[1, 2], keepdim=True)
        clip_idx = (deriv_norm > clipping).view(X.size(0))
        sigma_deriv[clip_idx] = (sigma_deriv[clip_idx] / deriv_norm[clip_idx]) * clipping

        sigma_deriv_times_Wf = (sigma_deriv.matmul(W_f.view(size=[S, 1])))  # N x S x 1
        return W_f, Wf_dot_sigma, sigma_deriv_times_Wf

    def hessian_G_f(self,
                    X: torch.Tensor,
                    W_f: torch.Tensor,
                    Wf_dot_sigma: torch.Tensor,
                    sigma_deriv_times_Wf: torch.Tensor):
        N = X.size(0)
        S = self.model.num_strains()

        h1 = -Wf_dot_sigma.pow(exponent=-2).expand([S, S, -1]).permute([2, 0, 1]) * (
            sigma_deriv_times_Wf.matmul(sigma_deriv_times_Wf.transpose(1, 2))  # N x S x S
        )

        sigma_second_deriv = self.sigmoid_hessian(X)  # N x S x S x S
        h2 = Wf_dot_sigma.reciprocal().expand([S, S, -1]).permute([2, 0, 1]) * (
            sigma_second_deriv.matmul(W_f.view(size=[S, 1])).view(N, S, S)
        )

        return h1 + h2  # N x S x S

    def grad_G_f(self, X: torch.Tensor, Wf_dot_sigma: torch.Tensor, sigma_deriv_times_Wf: torch.Tensor):
        N = X.size(0)
        S = self.model.num_strains()
        return Wf_dot_sigma.reciprocal().expand([S, -1]).t() * sigma_deriv_times_Wf.view(size=[N, S])  # N x S

    def VH_t(self, t: int, X_t: torch.Tensor, clipping=float("inf")):
        """
        Returns the pair V(X_t), H(X_t), the data-weighted sigmoid gradients and Hessians.

        :param t: the time index (for looking up read likelihoods)
        :param X_t: the Gaussian (an N x S tensor, rows indexed over samples).
        :param clipping: A gradient clipping threshold (in frobenius norm).
        :return: V (an N x S tensor) and H (an N x S x S tensor).
        """
        V = torch.zeros(X_t.size(0), X_t.size(1), device=cfg.torch_cfg.device)
        H = torch.zeros(X_t.size(0), X_t.size(1), X_t.size(1), device=cfg.torch_cfg.device)
        for f in range(self.model.num_fragments()):
            # TODO optimize these operations (they are extremely slow).
            W_f, Wf_dot_sigma, sigma_deriv_times_Wf = self.deriv_precomputation(f, X_t, clipping=clipping)
            phi_sum = self.phi[t][f].sum()
            H = H + self.hessian_G_f(X_t, W_f, Wf_dot_sigma, sigma_deriv_times_Wf) * phi_sum
            V = V + self.grad_G_f(X_t, Wf_dot_sigma, sigma_deriv_times_Wf) * phi_sum
        return V, H  # (N x S) and (N x S x S)

    def update(self) -> float:
        diff = 0.

        # for resampling.
        X_prev = self.model.mu
        log_w_prev = torch.zeros(size=[self.num_update_samples], device=cfg.torch_cfg.device)

        # Get the samples.
        for t in range(self.model.num_times()):
            if t == 0:
                X, proposal_log_likelihoods = self.sample_t0(
                    num_samples=self.num_update_samples
                )
            else:
                X, proposal_log_likelihoods = self.sample_t(
                    t=t,
                    X_prev=X_prev
                )
            # diff += self.update_t(t=t, X_t=X, X_prev=X_prev, proposal_log_likelihoods=proposal_log_likelihoods)

            actual_log_likelihoods = self.actual_log_likelihood(t, X, X_prev)  # length N
            log_w = log_w_prev + actual_log_likelihoods - proposal_log_likelihoods  # length N
            W_normalized = (log_w - log_w.max()).exp()  # length N
            W_normalized = W_normalized / W_normalized.sum()  # length N, normalized.

            # ESS criterion for resampling.
            N = self.num_update_samples
            do_resample = W_normalized.pow(2).sum().reciprocal().item() < (N / 2)
            if do_resample:
                children = Categorical(probs=W_normalized).sample([N])
                X_prev = X[children]
                W_normalized = (1 / N) * torch.ones(size=[N], device=cfg.torch_cfg.device)
                log_w_prev = W_normalized.log()
            else:
                log_w_prev = log_w

            diff += self.update_t(t=t, X_t=X,  weights=W_normalized)
        return diff

    def actual_log_likelihood(self, t, X_t, X_prev):
        return (
                (1 / self.model.time_scale(t) ** 2) * (X_t - X_prev).norm(p='fro', dim=1)
                +
                (softmax(X_t, dim=1).mm(self.model.get_fragment_frequencies().t())).log().mv(
                    self.phi[t].sum(dim=1)
                )
        )  # length N

    def update_t(self,
                 t: int,
                 X_t: torch.Tensor,
                 weights: torch.Tensor) -> float:
        """
        Update the model paramters (phi) from sequential importance samples.

        :param t: the time index of current sample.
        :param X_t: (N x S) tensor of proposal samples.
        :param weights: a length N vector of sequential sample weights.
        :return the difference (in L2 norm) of phi.
        """
        N = X_t.size(0)
        log_statistic = (softmax(X_t, dim=1)
                         .mm(self.model.get_fragment_frequencies().t())
                         .log())  # N x F
        tilt = (weights.view(1, N).mm(log_statistic)
                .exp()
                .view(size=[1, self.model.num_fragments()]))  # 1 x F tensor

        updated_phi_t = tilt.expand([self.read_counts[t], -1]).t() * self.read_likelihoods[t]
        updated_phi_t = updated_phi_t / updated_phi_t.sum(dim=0, keepdim=True)  # normalize.
        diff = (self.phi[t] - updated_phi_t).norm(p=2).item()
        self.phi[t] = updated_phi_t
        return diff

    def sample(self, num_samples=1,
               rejection_constant_init_log: float = -float("inf"),
               burnin: int = 2) -> List[torch.Tensor]:
        """
        Perform rejection sampling.
        :param burnin:
        :param rejection_constant_init_log:
        :param num_samples:
        """
        logger.debug("Sampling from mean-field posterior. "
                     "Using rejection sampling (may take a while; try lowering rejection_constant_upper_bound.).")
        S = self.model.num_strains()
        X = []
        for t in range(self.model.num_times()):
            x_t = torch.empty(size=[num_samples, S], device=cfg.torch_cfg.device)
            for i in range(num_samples):
                c = rejection_constant_init_log
                logger.debug("Sample \# {i}".format(i=i))
                x = None
                reject = True
                initial_samples = 0
                while reject or initial_samples < burnin:
                    if t == 0:
                        x_prev = self.model.mu
                        x, proposal_log_likelihood = self.sample_t0(num_samples=1)
                    else:
                        x_prev = X[t-1][i].view(1, S)
                        x, proposal_log_likelihood = self.sample_t(t=t, X_prev=x_prev)

                    log_ll_diff = (self.actual_log_likelihood(t, X_t=x, X_prev=x_prev) - proposal_log_likelihood).item()
                    # print("log ll diff: ", log_ll_diff)
                    thresh = log_ll_diff - c
                    # print("thresh = ", thresh)
                    reject = torch.rand(1, device=cfg.torch_cfg.device).log().item() > thresh
                    initial_samples = initial_samples + (0 if reject else 1)
                    # print("reject = ", reject, " initial_samples = ", initial_samples)
                    c = max(c, log_ll_diff)
                    # print("new constant = ", c)
                x_t[i] = x
            X.append(x_t)
        return X

    def sample_t0(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        S = self.model.num_strains()
        center = self.mu
        V, H = self.VH_t(t=0, X_t=center.view(size=[1, S]), clipping=self.clipping)
        precision = self.stdev_scale[0] * torch.eye(S, device=cfg.torch_cfg.device) / (self.model.time_scale(0) ** 2) - H.view(S, S)
        loc = precision.inverse().matmul(V.view(S, 1)).view(size=[S]) + self.mu

        try:
            dist_0 = MultivariateNormal(
                loc=loc,
                precision_matrix=precision,
            )
        except Exception as e:
            print("Gaussian precision matrix not positive definite. Time t = {}".format(0))
            raise e

        samples = dist_0.sample(sample_shape=[num_samples])
        likelihoods = dist_0.log_prob(samples)
        return samples, likelihoods

    def sample_t(self, t: int, X_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = X_prev.size(0)
        S = self.model.num_strains()
        V, H = self.VH_t(t=t, X_t=X_prev, clipping=self.clipping)

        # N x S x S
        precision = self.stdev_scale[t] * torch.eye(S, device=cfg.torch_cfg.device).expand([N, -1, -1]) \
                    / (self.model.time_scale(t) ** 2) \
                    - H

        loc = X_prev + precision.inverse().matmul(
            V.view(size=[N, S, 1])
        ).view(size=[N, S])

        try:
            dist = MultivariateNormal(
                loc=loc,
                precision_matrix=precision,
            )
        except Exception as e:
            logger.error("Gaussian precision matrix not positive definite. Time t = {}".format(t))
            raise e


        samples = dist.sample()
        likelihoods = dist.log_prob(samples)
        return samples, likelihoods

    def sigmoid_derivative(self, X: torch.Tensor) -> torch.Tensor:
        N = X.size(0)
        S = self.model.num_strains()
        sigmoid = softmax(X, dim=1).view(N, S, 1)
        deriv = sigmoid.matmul(sigmoid.transpose(1, 2))
        for n in range(N):
            deriv[n] = torch.diag(sigmoid[n].view(S)) - deriv[n]
        return deriv[:, :-1, :]  # N x S x S

    def sigmoid_hessian(self, X: torch.Tensor) -> torch.Tensor:  # N x S x S x S
        """
        The formula is:
        dS_k / (dx_i dx_j)
        = T_{i,j,k}
        = 2 S_i S_j S_k - delta_{ij} S_i S_k - delta_{ik} S_j S_k - delta_{jk} S_i S_k + delta_{ijk} S_k

        which gives T = 2(S x S x S)
                        - (S x diag(S)).permute(0,1,2)
                        - (S x diag(S)).permute(1,2,0)
                        - (S x diag(S)).permute(2,0,1)
                        + Diag_3(S)
        (where "x" is tensor product)
        """
        N = X.size(0)
        S = self.model.num_strains()
        sigmoid = softmax(X, dim=1).view(N, S, 1)
        hess = sigmoid.matmul(sigmoid.transpose(1, 2)).view(N, S, S, 1).matmul(
            sigmoid.expand([S, -1, -1, -1]).permute([1, 0, 3, 2])
        )  # N x S x S x S
        for n in range(N):
            cross = (torch.diag(sigmoid[n].view(S))
                     .view(S, S, 1)
                     .matmul(sigmoid[n].t())
                     )
            hess[n] = 2*hess[n] - cross.permute([0, 1, 2]) - cross.permute([1, 2, 0]) - cross.permute([2, 0, 1])
            for i in range(S):
                hess[n, i, i, i] = hess[n, i, i, i] + sigmoid[n, i, 0]
        return hess


class SecondOrderVariationalSolver(AbstractModelSolver):
    """
    The VI formulation based on the second-order Taylor approximation (Hessian calculation).
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 cache_tag: str):
        super().__init__(model, data, cache_tag)

    def solve(self,
              iters=4000,
              num_montecarlo_samples=1000,
              print_debug_every=200,
              thresh=1e-5,
              clipping=50.,
              stdev_scale=1.):
        posterior = MeanFieldPosterior(
            model=self.model,
            read_counts=[len(reads) for reads in self.data],
            num_update_samples=num_montecarlo_samples,
            read_likelihoods=self.read_likelihoods,
            device=cfg.torch_cfg.device,
            clipping=clipping,
            stdev_scale=stdev_scale
        )

        logger.debug("Variational Inference algorithm started. (Second-order heuristic, Target iterations={it})".format(
            it=iters
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        for i in range(1, iters+1, 1):
            time_est.stopwatch_click()
            last_diff = posterior.update()  # <------ VI update step.
            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            has_converged = (last_diff < thresh)
            if has_converged:
                logger.info("Convergence criterion ({th}) met; terminating optimization early.".format(th=thresh))
                break

            if i % print_debug_every == 0:
                logger.info("Iteration {i} "
                            "| time left: {t:.2f} min. "
                            "| Last param diff: {diff}"
                            .format(i=i,
                                    t=time_est.time_left() / 60000,
                                    diff=last_diff)
                            )

        return posterior

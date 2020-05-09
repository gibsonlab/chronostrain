from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
from torch.distributions import MultivariateNormal, Categorical

from model.generative import GenerativeModel
from model.reads import SequenceRead
from algs.base import AbstractModelSolver, compute_read_likelihoods

from util.torch import multi_logit
from util.benchmarking import RuntimeEstimator
from util.io.logger import logger


class AbstractVariationalPosterior(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, num_samples=1) -> List[torch.Tensor]:
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
                 device):
        self.model = model
        self.read_counts = read_counts
        self.device = device
        self.num_update_samples = num_update_samples
        self.read_likelihoods = read_likelihoods

        # Variational parameters
        self.mu = torch.zeros(size=[model.num_strains()-1], device=self.device)
        self.phi = [
            (1 / model.num_fragments()) * torch.ones(size=[model.num_fragments(), read_counts[t]], device=self.device)
            for t in range(model.num_times())
        ]  # each is an F x R_t tensor.

    def deriv_precomputation(self, f: int, X: torch.Tensor):
        S = self.model.num_strains()

        W_f = self.model.get_fragment_frequencies()[f]  # S-dim vector.
        sigma = multi_logit(X, dim=1)  # N x S
        Wf_dot_sigma = sigma.mv(W_f)  # N-dim vector.

        sigma_deriv = self.sigmoid_derivative(X)  # N x S-1 x S
        sigma_deriv_times_Wf = (sigma_deriv.matmul(W_f.view(size=[S, 1])))  # N x S-1 x 1
        return W_f, Wf_dot_sigma, sigma_deriv_times_Wf

    def hessian_G_f(self,
                    X: torch.Tensor,
                    W_f: torch.Tensor,
                    Wf_dot_sigma: torch.Tensor,
                    sigma_deriv_times_Wf: torch.Tensor):
        N = X.size(0)
        S = self.model.num_strains()

        h1 = -Wf_dot_sigma.pow(exponent=-2).expand([S-1, S-1, -1]).permute([2, 0, 1]) * (
            sigma_deriv_times_Wf.matmul(sigma_deriv_times_Wf.transpose(1, 2))  # N x S-1 x S-1
        )

        sigma_second_deriv = self.sigmoid_hessian(X)  # N x (S-1) x (S-1) x S
        h2 = Wf_dot_sigma.reciprocal().expand([S-1, S-1, -1]).permute([2, 0, 1]) * (
            sigma_second_deriv.matmul(W_f.view(size=[S, 1])).view(N, S-1, S-1)
        )

        return h1 + h2  # N x S-1 x S-1

    def grad_G_f(self, X: torch.Tensor, Wf_dot_sigma: torch.Tensor, sigma_deriv_times_Wf: torch.Tensor):
        N = X.size(0)
        S = self.model.num_strains()
        return Wf_dot_sigma.reciprocal().expand([S-1, -1]).t() * sigma_deriv_times_Wf.view(size=[N, S-1])  # N x S-1

    def VH_t(self, t: int, X_t: torch.Tensor):
        """
        Returns the pair V(X_t), H(X_t), the data-weighted sigmoid gradients and Hessians.

        :param t: the time index (for looking up read likelihoods)
        :param X_t: the Gaussian (an N x S-1 tensor, rows indexed over samples).
        :return: V (an N x S-1 tensor) and H (an N x S-1 x S-1 tensor).
        """
        V = torch.zeros(X_t.size(0), X_t.size(1), device=self.device)
        H = torch.zeros(X_t.size(0), X_t.size(1), X_t.size(1), device=self.device)
        for f in range(self.model.num_fragments()):
            # TODO optimize these operations (they are extremely slow).
            W_f, Wf_dot_sigma, sigma_deriv_times_Wf = self.deriv_precomputation(f, X_t)
            phi_sum = self.phi[t][f].sum()
            H = H + self.hessian_G_f(X_t, W_f, Wf_dot_sigma, sigma_deriv_times_Wf) * phi_sum
            V = V + self.grad_G_f(X_t, Wf_dot_sigma, sigma_deriv_times_Wf) * phi_sum
        return V, H  # (N x S-1) and (N x S-1 x S-1)

    def update(self, resample=True) -> float:
        diff = 0.

        # for resampling.
        X_prev = None

        # Get the samples.
        for t in range(self.model.num_times()):
            if t == 0:
                print("Sampling t0")
                X, log_likelihoods = self.sample_t0(num_samples=self.num_update_samples)
            else:
                X, log_likelihoods = self.sample_t(t=t, X_prev=X_prev)
            diff += self.update_t(t=t, X_t=X)

            if resample:
                logW = self.model.log_likelihood_xt(
                    t=t, X=X, X_prev=X_prev, read_likelihoods=self.read_likelihoods[t]
                ) - log_likelihoods
                W_normalized = (logW - logW.max()).exp()
                W_normalized = W_normalized / W_normalized.sum()
                children = Categorical(probs=W_normalized).sample([self.num_update_samples])
                X_prev = X[children]
            else:
                X_prev = X
        return diff

    def update_t(self, t, X_t) -> float:
        estimated_tilt = (multi_logit(X_t, dim=1)
                          .mm(self.model.get_fragment_frequencies().t())
                          .log()  # N x F
                          .mean(dim=0)
                          .exp()
                          .view(size=[1, self.model.num_fragments()]))  # (1 x F) tensor
        updated_phi_t = estimated_tilt.expand([self.read_counts[t], -1]).t() * self.read_likelihoods[t]
        updated_phi_t = updated_phi_t / updated_phi_t.sum(dim=0, keepdim=True)  # normalize.
        diff = (self.phi[t] - updated_phi_t).norm(p=2).item()
        self.phi[t] = updated_phi_t
        return diff

    def sample(self, num_samples=1) -> List[torch.Tensor]:
        X = []
        for t in range(self.model.num_times()):
            if t == 0:
                sample, _ = self.sample_t0(num_samples=num_samples)
                X.append(sample)
            else:
                sample, _ = self.sample_t(t=t, X_prev=X[t-1])
                X.append(sample)
        return X

    def sample_t0(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        S = self.model.num_strains()
        center = self.mu
        V, H = self.VH_t(t=0, X_t=center.view(size=[1, S - 1]))
        precision = torch.eye(S - 1, device=self.device) / (self.model.time_scale(0) ** 2) - H.view(S - 1, S - 1)
        loc = precision.inverse().matmul(V.view(S - 1, 1)).view(size=[S - 1]) + self.mu

        dist_0 = MultivariateNormal(
            loc=loc,
            precision_matrix=precision,
        )
        samples = dist_0.sample(sample_shape=[num_samples])
        likelihoods = dist_0.log_prob(samples)
        return samples, likelihoods

    def sample_t(self, t: int, X_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = X_prev.size(0)
        S = self.model.num_strains()
        V, H = self.VH_t(t=t, X_t=X_prev)

        # N x S-1 x S-1
        precision = torch.eye(S-1, device=self.device).expand([N, -1, -1]) \
                    / (self.model.time_scale(t) ** 2) \
                    - H
        loc = X_prev + precision.inverse().matmul(
            V.view(size=[N, S-1, 1])
        ).view(size=[N, S-1])

        # print("V = ", V)
        # print("loc = ", loc)
        # print("covar = ", precision.inverse())

        dist = MultivariateNormal(
            loc=loc,
            precision_matrix=precision,
        )

        samples = dist.sample()
        likelihoods = dist.log_prob(samples)
        return samples, likelihoods

    def sigmoid_derivative(self, X: torch.Tensor) -> torch.Tensor:
        N = X.size(0)
        S = self.model.num_strains()
        sigmoid = multi_logit(X, dim=1).view(N, S, 1)
        deriv = sigmoid.matmul(sigmoid.transpose(1, 2))
        for n in range(N):
            deriv[n] = torch.diag(sigmoid[n].view(S)) - deriv[n]
        return deriv[:, :-1, :]  # N x S-1 x S

    def sigmoid_hessian(self, X: torch.Tensor) -> torch.Tensor:  # N x (S-1) x (S-1) x S
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
        sigmoid = multi_logit(X, dim=1).view(N, S, 1)
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
        return hess[:, :-1, :-1, :]  # cut off the last variable for multi_logit


class SecondOrderVariationalSolver(AbstractModelSolver):
    """
    The VI formulation based on the second-order Taylor approximation (Hessian calculation).
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 torch_device):
        super().__init__(model, data)
        self.device = torch_device
        self.read_likelihoods = compute_read_likelihoods(model=model, reads=data, logarithm=False, device=torch_device)

    def solve(self,
              iters=4000,
              num_montecarlo_samples=1000,
              print_debug_every=200,
              thresh=1e-5,
              do_resampling=True):
        posterior = MeanFieldPosterior(
            model=self.model,
            read_counts=[len(reads) for reads in self.data],
            num_update_samples=num_montecarlo_samples,
            read_likelihoods=self.read_likelihoods,
            device=self.device
        )

        logger.debug("Variational Inference algorithm started. (Second-order heuristic, Target iterations={it})".format(
            it=iters
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        for i in range(1, iters+1, 1):
            time_est.stopwatch_click()
            last_diff = posterior.update(resample=do_resampling)  # <------ VI update step.
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

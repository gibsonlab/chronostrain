"""
 model_solver.py
 Contains implementations of the proposed algorithms.
"""

import math
import numpy as np
from algs import em


# Helper function for both EM and VI
def compute_frag_errors(model, reads):
    return [
        [
            [np.exp(model.error_model.compute_log_likelihood(f, r)) for f in model.fragment_space]
            for r in reads[k]
        ]
        for k in range(len(model.times))
    ]

# ===========================================================================================
# =============== Expectation-Maximization (for getting a MAP estimator) ====================
# ===========================================================================================


def em_estimate(model, reads, tol=1e-10, iters=10000):
    """
    :param model: A GenerativeModel instance.
    :param reads: A time-indexed list of read sets. Each entry is itself a list of reads for time t.
    :param tol: the convergence tolerance threshold for the objective.
    :param iters: Number of iterations to stop after if no convergence has been achieved yet.
    :return: Output the learned MAP estimator for the strain trajectory
    """
    # TODO: Use tolerance termination condition with respect to change in P(X | data)

    print("Running EM algorithm")

    # Initialization
    frag_errors = compute_frag_errors(model, reads)
    means = np.zeros((len(model.times), len(model.bacteria_pop.strains)))

    # Update
    step_size = .0001
    prev_ssn = float('inf')
    ssn_change = None
    iter_num = 0

    while ssn_change is None or (ssn_change >= tol and iter_num < iters):
        iter_num += 1

        if iter_num % 500 == 0:
            print("Iteration: ", iter_num)

        em.em_update(model, reads, means, frag_errors, step_size)


        ssn = sum([np.linalg.norm(vec, 2)**2 for vec in means])  # Sum of squared norms.
        ssn_change = np.abs(prev_ssn - ssn)
        prev_ssn = ssn

    return means  # TODO: return frag_freqs? Like the VI alg does.


# ================================================================================================
# =============== Variational Inference (for learning approximate posteriors) ====================
# ================================================================================================


def variational_learn(model, reads, tol=1e-5):
    """
    :param model: A GenerativeModel instance.
    :param reads: A time-indexed list of read sets. Each entry is itself a list of reads for time t.
    :param tol: the convergence tolerance threshold for the objective.
    :return: Output the learned parameters for the mean-field posterior.
    """
    # Initialization
    frag_errors = compute_frag_errors(model, reads)
    means = [np.zeros(model.strains, 1) for t in model.times]
    covariances = [model.time_scale(k) * np.eye(model.strains) for k in range(model.num_times())]
    frag_freqs = [
        (1 / len(model.fragment_space)) * np.ones(len(model.fragment_space), len(reads[t]))
        for t in model.times
    ]

    # Update
    prev_loss = None
    loss_change = None
    while (loss_change is None or loss_change < tol):
        loss = variational_update(model, reads, means, covariances, frag_freqs, frag_errors)
        loss_change = np.abs(prev_loss - loss)
        prev_loss = loss

    return means, covariances, frag_freqs


def compute_model_ELBO(model, reads, means, covariances, frag_freqs):
    raise NotImplementedError()


def variational_update(model, reads, means, covariances, frag_freqs, frag_errors):
    # ==== update gaussians.
    prev_mean = model.mu
    for t_idx in range(len(model.times())):
        frag_probs = np.sum(frag_freqs[t_idx], axis=1)  # sum the frequencies over reads (sum_i \phi^{t,i}_{f})
        (H, V) = gradient_update(model, frag_probs, prev_mean)
        covariances[t_idx] = np.linalg.inv((1 / math.pow(model.time_scale(t_idx), 2)) * np.eye(model.num_strains()) - H)
        means[t_idx] = prev_mean + np.matmul(covariances[t_idx], V)

    # ==== update fragment probabilities.
    for t_idx in range(len(model.times())):
        E_log_z = log_frequency_expectations(model, means, covariances, t_idx)
        for r_idx in range(len(reads)):
            frag_freqs[t_idx][r_idx, :] = frag_errors[t_idx][r_idx] * np.exp(E_log_z)

    # ==== Return ELBO loss.
    return compute_model_ELBO(model, reads, means, covariances, frag_freqs)


def log_frequency_expectations(model, means, covariances, t_idx):
    # Mathematically non-rigorous. Only should work if covariances are very tiny, e.g. O(1/sqrt(N)).
    return np.log(model.W * softmax(means[t_idx]))


def gradient_update(model, frag_probs, center):
    H = np.zeros(model.num_strains(), model.num_strains)
    V = np.zeros(model.num_strains(), 1)
    for frag in range(len(model.fragment_space)):
        H_f = frag_probs[frag] * map_hessian(center, model.W, frag)
        V_f = frag_probs[frag] * np.transpose(map_gradient(center, model.W, frag))
        H = H + H_f
        V = V + V_f
    return (H, V)


def map_gradient(center, W, f):
    # Outputs a row vector.
    deriv = np.matmul(W[f, :]) * softmax_derivative(x=center)
    return deriv / np.matmul(W[f, :], softmax(x=center))


def map_hessian(center, W, f):
    N = len(center)
    dot_product = np.matmul(W[f, :], softmax(x=center))
    d = map_gradient(center, W, f)
    tensor = softmax_second_derivative_tensor(center)
    second_deriv = np.zeros((N, N))
    for k in range(N):
        second_deriv = second_deriv + (W[f, k] * tensor[k, :, :])
    return (second_deriv / dot_product) - np.matmul(np.transpose(d), d)


def softmax_derivative(x):
    s = softmax(x)
    N = len(s)
    deriv = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            deriv[i][j] = s[i] * (delta(i, j) - s[j])
    return deriv
    # sbar = 1 - s
    # return s * np.transpose(sbar) - (np.ones((N, N)) - np.eye(N)) * s[:, None]


def softmax_second_derivative_tensor(x):
    s = softmax(x)
    N = len(s)
    deriv = np.zeros((N, N, N))
    for k in range(N):
        for i in range(N):
            for j in range(N):
                # second derivative of sigma_k (with respect to x_i, x_j)
                deriv[s][i][j] = s[k] * (
                        ((delta(j, k) - s[j]) * (delta(i, k) - s[i]))
                        -
                        (s[i] * (delta(i, j) - s[j]))
                )
    return deriv


def delta(i, j):
    if i == j:
        return 1
    return 0


def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

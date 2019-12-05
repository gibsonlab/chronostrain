import cvxpy as cvx
from cvxpy.expressions.expression import Expression
import numpy as np
import scipy


# "design" matrix is the "A" matrix from the notes.
def qp_solve(times, design, observations, verbose=True, lasso_weight=0.0):
    num_times = len(times)
    inv_time_diffs = np.reciprocal(times[1:num_times] - times[0:num_times-1])
    abundances = cvx.Variable(
        shape=(np.size(design, 1), num_times)
    )

    #----------- OBJECTIVE
    objective = cvx.Minimize(
        cvx.norm(observations - design * abundances, "fro")
        + _col_diff_norms(abundances, inv_time_diffs)
        + lasso_weight * cvx.sum(cvx.norm(abundances, axis=0, p=1))
    )

    #----------- CONSTRAINTS
    constraints = [abundances >= np.zeros(shape=(np.size(design, 1), num_times))]

    #----------- Solve!
    problem = cvx.Problem(objective, constraints)

    # result = problem.solve(
    #     solver=cvx.CPLEX,
    #     verbose=True,
    #     cplex_params={
    #         'emphasis.numerical' : 1,
    #         'lpmethod' : 4,
    #         #'simplex.tolerances.optimaltiy' : 2e-9,
    #         #'barrier.convergencetol' : 2e-9,
    #     }
    # )

    # data = problem.get_problem_data(cvx.GUROBI)
    result = problem.solve(solver=cvx.GUROBI, verbose=verbose, Presolve=0, Method=2)

    return abundances.value


def _col_diff_norms(mat, row_scales):
    # remember that everything is transposed. need column diff norms.
    if mat.ndim != 2:
        raise ValueError("_col_diff_norms must take a matrix argument.")
    diffs = []
    i = 0
    for scale in row_scales:
        diffs += [scale * cvx.norm(mat[:, i+1] - mat[:, i], p=2)] # double-check this part
        i += 1
    return cvx.sum(diffs, "fro")


def least_squares_solve(times, design, observations):
    img = np.transpose(design) * observations
    out = np.array(design.size(1), len(times))

    xform = np.matmul(np.transpose(design), design)

    delta = 1 / (times[1] - times[0])
    out[:, 0] = (xform.dot(img[:, 0]) - xform.dot(img[:, 1])) * delta

    delta = 1 / (times[-1] - times[-2])
    out[:, -1] = (xform.dot(img[:, -1]) - xform.dot(img[:, -2])) * delta

    for t in range(1, len(times)-1):
        delta1 = 1 / (times[t] - times[t-1])
        delta2 = 1 / (times[t+1] - times[t])
        out[:, t] = -xform.dot(img[:, t-1]) * delta1 + xform.dot(img[:, t]) * delta1 * delta2 - xform.dot(img[:, t+1]) * delta2





import numpy as np
from algs import qp_solve


def random_probability_vector(length):
    vec = np.random.uniform(low=0.01, high=1.0, size=length)
    vec = vec / np.sum(vec)
    return vec


def random_vector(size):
    vec = np.random.uniform(low=10.0, high=1000.0, size=size)
    return vec


def unique_designs(design):
    """
    Check whether columns are unique.
    :param design:
    :return:
    """
    return np.unique(design, axis=1)


def generate_abundances(times, num_strains, scale=1.0, sparsify_to=None):
    abundances = []
    previous_t = None
    for t in times:
        if len(abundances) == 0:
            abundances.append(random_vector(num_strains))
        else:
            next_value = np.random.multivariate_normal(
                mean=abundances[-1],
                cov=np.eye(num_strains) * scale * (t - previous_t),
            )
            abundances.append(next_value)
        previous_t = t
    abundances = np.transpose(np.stack(abundances))
    if sparsify_to is not None:
        choices = np.random.choice(num_strains, num_strains - sparsify_to, replace=False)
        abundances[choices, :] = np.zeros(shape=[num_strains - sparsify_to, len(times)])
    return abundances


times = np.array([t for t in range(1,20)])
num_strains = 40
num_genotypes = 20


# (Strains X Genotypes)
count_high = 2
design = np.random.randint(low=0, high=count_high, size=(num_genotypes, num_strains), dtype='int64')
#design = unique_designs(design)

# Gaussian process of relative abundances
# (strains X times)
abundances = generate_abundances(times, num_strains, scale=0.5, sparsify_to=15)

# (Genotypes X times)
observations = np.dot(design, abundances)


qp_result = qp_solve(times, design, observations, verbose=False, lasso_weight=0.0)
qp_result_lasso = qp_solve(times, design, observations, verbose=False, lasso_weight=1.0)

print(design)
print(abundances)

print("RESULT:")
# print(qp_result)
print(np.mean(np.abs(abundances - qp_result)))

print("RESULT (with lasso):")
# print(qp_result_lasso)
print(np.mean(np.abs(abundances - qp_result_lasso)))
import random
import numpy as np
from algs import em
from model import bacteria, reads, generative

random.seed(123)

##############################
# Generate bacteria population
num_strains = 4
num_markers = 1

marker_length = 150

strains = []
for i in range(num_strains):

    markers = []
    for j in range(num_markers):
        nucleotides = "".join([random.choice(["A", "G", "C", "T"]) for i in range(marker_length)])
        new_marker = bacteria.Marker(nucleotides)
        markers.append(new_marker)

    new_strain = bacteria.Strain(markers)
    strains.append(new_strain)

my_bacteria_pop = bacteria.Population(strains)
##############################

# Construct generative model
times = [1, 2, 3]
mu = np.array([0] * len(my_bacteria_pop.strains))  # One dimension for each strain
tau_1 = 1
tau = 1
read_length = 100

my_error_model = reads.BasicErrorModel(read_len=read_length)

my_model = generative.GenerativeModel(times=times,
                                      mu=mu,
                                      tau_1=tau_1,
                                      tau=tau,
                                      bacteria_pop=my_bacteria_pop,
                                      read_length=read_length,
                                      read_error_model=my_error_model)
print("Constructed model!")

num_samples = [1, 5, 4]

print("==================")
print("Sampling reads and abundances...")
sampled_abundances, sampled_reads = my_model.sample_abundances_and_reads(num_samples=num_samples)

print(sampled_reads)

print("Sampled abundances:")
print(sampled_abundances)


my_em_solver = em.EMSolver(my_model, sampled_reads)
print(my_em_solver.solve())


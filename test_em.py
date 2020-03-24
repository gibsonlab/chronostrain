import random
import numpy as np
from algs import model_solver
from model import bacteria, reads, generative

random.seed(123)

# Generate bacteria population
num_strains = 4
num_markers = 1
fragment_length = 10

my_bacteria_pop = bacteria.Population(num_strains=num_strains,
                                      num_markers=num_markers,
                                      marker_length=fragment_length * 300,
                                      num_snps=(fragment_length * 300) // 100)

# Construct generative model
times = [1, 2, 3]
mu = np.array([0] * my_bacteria_pop.num_strains)  # One dimension for each strain
tau_1 = 1
tau = 1
W = my_bacteria_pop.get_fragment_space(window_size=fragment_length)
strains = my_bacteria_pop.strains
fragment_space = my_bacteria_pop.get_fragment_space(window_size=fragment_length)
print("Successfully constructed fragment space!")

read_error_model = reads.BasicErrorModel(read_len=fragment_length)
bacteria_pop = my_bacteria_pop

my_model = generative.GenerativeModel(times=times,
                                      mu=mu,
                                      tau_1=tau_1,
                                      tau=tau,
                                      W=W,
                                      fragment_space=fragment_space,
                                      read_error_model=read_error_model,
                                      bacteria_pop=bacteria_pop)
print("Constructed model!")

num_samples = [1, 5, 4]

print("==================")
print("Sampling reads and abundances...")
sampled_abundances, sampled_reads = my_model.sample_abundances_and_reads(num_samples=num_samples)

print("Sampled abundances:")
print(sampled_abundances)
print("Completed!")

print(model_solver.em_estimate(my_model, sampled_reads))

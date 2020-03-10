import numpy as np
from model import reads
from model import generative
from model import bacteria
import random


def mean_squared_error(brownian_motion, brownian_motion_guess):
    assert brownian_motion.shape == brownian_motion_guess.shape

    total = 0
    for row, row_guess in zip(brownian_motion, brownian_motion_guess):
        for ele, ele_guess in zip(row, row_guess):
            total += (ele - ele_guess) ** 2

    return total / np.prod(brownian_motion.shape)


def em_iterate(brownian_motion_guess, e_matrix, step_size):

    rel_abundances_motion_guess = my_model.generate_relative_abundances(brownian_motion_guess)
    updated_brownian_motion_guess = []

    for time_index, (abundances_at_t, guessed_abundances_at_t, sampled_reads_at_t) in \
            enumerate(zip(sampled_abundances, rel_abundances_motion_guess, sampled_reads)):
        # For Debugging
        # time_index = 0
        # abundances_at_t = sampled_abundances[time_index]
        # guessed_abundances_at_t = rel_abundances_motion_guess[time_index]
        # sampled_reads_at_t = sampled_reads[time_index]

        ##############################
        # Compute the "Q" vector
        ##############################

        time_indexed_fragment_frequencies_guess = my_model.generate_time_indexed_fragment_frequencies(
            rel_abundances_motion_guess[time_index])

        # Step 1
        v = []
        for read_index, read in enumerate(sampled_reads_at_t):
            read_v = np.multiply(e_matrix[time_index][read_index], time_indexed_fragment_frequencies_guess)
            read_v = read_v / sum(read_v)
            v.append(read_v)

        # Step 2
        v = np.asanyarray(v)
        v = sum(v)

        # Step 3
        q_guess = np.divide(v, time_indexed_fragment_frequencies_guess)

        ##############################
        # Compute the regularization term
        ##############################

        if time_index == 1:
            regularization_term = brownian_motion_guess[time_index] - brownian_motion_guess[time_index + 1]
        elif time_index == len(my_model.times) - 1:
            regularization_term = brownian_motion_guess[time_index] - brownian_motion_guess[time_index - 1]
        else:
            regularization_term = (2 * brownian_motion_guess[time_index] -
                                   brownian_motion_guess[time_index - 1] -
                                   brownian_motion_guess[time_index + 1])

        scaled_tau = my_model.time_scale(time_index) ** 2
        regularization_term *= -1 / scaled_tau

        ##############################
        # Compute the derivative of relative abundances at X^t
        # An S x S Jacobian
        ##############################

        sigma_prime = np.zeros((len(guessed_abundances_at_t), len(guessed_abundances_at_t)))
        for i in range(len(guessed_abundances_at_t)):
            for j in range(len(guessed_abundances_at_t)):
                delta = 1 if i == j else 0
                sigma_prime[i][j] = guessed_abundances_at_t[i] * (delta - guessed_abundances_at_t[j])

        assert sigma_prime.shape == (my_model.num_strains(), my_model.num_strains())

        ##############################
        # Compute the 'main' term
        ##############################

        W = my_model.strain_fragment_matrix

        main_term = np.matmul(np.matmul(np.transpose(sigma_prime), np.transpose(W)), q_guess)

        ##############################
        # Update our guess for the motion at this time step
        ##############################

        updated_brownian_motion_guess.append(
            brownian_motion_guess[time_index] + step_size * (main_term + regularization_term)
        )

    return np.asanyarray(updated_brownian_motion_guess)


def em_run(model, sampled_abundances, sampled_reads, step_size=0.001, num_iterations=50):

    errorModel = reads.BasicErrorModel()

    ########################################
    # For each time step, for each read at that time, calculate the probability of that read condition on each fragment

    e_matrix = []
    for time_index, (abundances_at_t, sampled_reads_at_t) in enumerate(zip(sampled_abundances, sampled_reads)):

        e_t_vector = np.zeros((len(sampled_reads_at_t), len(model.fragment_space)))
        for read_index, read in enumerate(sampled_reads_at_t):

            e_i_t = np.zeros(len(model.fragment_space))
            for fragment_index, fragment in enumerate(model.fragment_space):

                # probability of reading fragment as read_i
                p = np.exp(errorModel.compute_log_likelihood(fragment=fragment, read=read))
                e_i_t[fragment_index] = p

            e_t_vector[read_index] = e_i_t
        e_matrix.append(e_t_vector)
    e_matrix = np.asanyarray(e_matrix)

    ########################################
    # Initialize guess of the trajectory X

    brownian_motion_guess = np.zeros((len(my_model.times), my_model.bacteria_pop.num_strains))

    ########################################
    # Run EM algorithm

    print("Initialization; Mean squared error: ", mean_squared_error(my_model.brownian_motion, brownian_motion_guess))
    for i in range(num_iterations):
        updated_brownian_motion_guess = em_iterate(brownian_motion_guess, e_matrix, step_size)
        brownian_motion_guess = updated_brownian_motion_guess
        print("Iteration: " + str(i) + "; Mean squared error: " + str(mean_squared_error(my_model.brownian_motion,
                                                                                         brownian_motion_guess)))

    print(brownian_motion_guess)


if __name__ == "__main__":
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
    times = [1, 2, 4]
    mu = np.array([0] * my_bacteria_pop.num_strains) # One dimension for each strain
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

    num_samples = [5, 6, 5]
    print("Sampling reads...")
    sampled_reads = my_model.sample_reads(num_samples=num_samples)
    print("Sampled reads:")
    print(sampled_reads)

    print("==================")
    print("Sampling reads and abundances...")
    sampled_abundances, sampled_reads = my_model.sample_abundances(num_samples=num_samples)

    print("Sampled abundances:")
    print(sampled_abundances)
    print("Completed!")
    em_run(my_model, sampled_abundances, sampled_reads)

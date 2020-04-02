import numpy as np


def em_update(model, reads, means, frag_errors, step_size):

    updated_means = []
    rel_abundances_motion_guess = model.generate_relative_abundances(means)

    for time_index, (guessed_abundances_at_t, reads_at_t) in \
            enumerate(zip(rel_abundances_motion_guess, reads)):

        ##############################
        # Compute the "Q" vector
        ##############################

        time_indexed_fragment_frequencies_guess = model.generate_time_indexed_fragment_frequencies(
            rel_abundances_motion_guess[time_index])

        # Step 1
        v = []
        for read_index, read in enumerate(reads_at_t):
            read_v = np.multiply(frag_errors[time_index][read_index], time_indexed_fragment_frequencies_guess)
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
            regularization_term = means[time_index] - means[time_index + 1]
        elif time_index == len(model.times) - 1:
            regularization_term = means[time_index] - means[time_index - 1]
        else:
            regularization_term = (2 * means[time_index] -
                                   means[time_index - 1] -
                                   means[time_index + 1])

        scaled_tau = model.time_scale(time_index) ** 2
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

        ##############################
        # Compute the 'main' term
        ##############################

        W = model.fragment_frequencies

        main_term = np.matmul(np.matmul(np.transpose(sigma_prime), np.transpose(W)), q_guess)

        ##############################
        # Update our guess for the motion at this time step
        ##############################

        updated_means.append(
            means[time_index] + step_size * (main_term + regularization_term)
        )

    for t_idx, new_mean in enumerate(updated_means):
        means[t_idx] = new_mean

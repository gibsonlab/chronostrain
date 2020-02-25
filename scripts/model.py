#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:41:50 2020

@author: Daniel Alfonsetti

@description:
    Generative model for time series bacterial strain tracking in metagenomic samples.
    
"""

import numpy as np
import random 
import copy
from enum import Enum


class Q_Score(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    TERRIBLE = 0


class Q_Score_Distribution():
    # Quality score distribution class
    
    def __init__(self, distribution = None):
        
        if distribution:
            self.distribution = distribution
            
        self.distribution = {Q_Score.HIGH : 0.5, 
                             Q_Score.MEDIUM : 0.3, 
                             Q_Score.LOW : 0.15, 
                             Q_Score.TERRIBLE: 0.05}
    
    def get_simple_q_score_vector(self, fragment):
        """
        Returns a quality score vector ('q vector') for the fragment
        
    
        The quality scores are chosen in proprotion to the distribution specified
        in self.distribution. 
        
        The current implementation assigns quality scores such that 1/2 of the specified
        frequency of the "terrible" score is applied to the first base pairs in the fragment
        as well as the last base pairs in the fragment.
        
        This pattern is reapeated with the 'low quality' score with the base pairs
        on both ends of the fragment that haven't been assigned. The resulting quality 
        pattern along the fragment is thus as follows:
            
            Terrible - Low - Medium - High - Medium - Low - Terrible
  

        @param - fragment --
            A list of characters (A, U, C, T)
        
        @return - 
            A list with length equal to the length of the input fragment list, and
            each element is an integer (0-3) representing a quality score.

        """
        
        # Implementation modified from 
        # https://stackoverflow.com/questions/32330459/partition-a-list-into-sublists-by-percentage
                
        
        percentages = [self.distribution[Q_Score.TERRIBLE]/2, 
                       self.distribution[Q_Score.LOW]/2, 
                       self.distribution[Q_Score.MEDIUM]/2, 
                       self.distribution[Q_Score.HIGH], 
                       self.distribution[Q_Score.MEDIUM]/2, 
                       self.distribution[Q_Score.LOW]/2, 
                       self.distribution[Q_Score.TERRIBLE]/2]
        
        splits = np.cumsum(percentages)

        if splits[-1] != 1:
            raise ValueError("percents don't add up to 100")
    
        # Split doesn't need last percent, it will just take what is left
        splits = splits[:-1]
    
        # Turn values into indices
        splits *= len(fragment)
    
        # Turn double indices into integers.
        # CAUTION: numpy rounds to closest EVEN number when a number is halfway
        # between two integers. So 0.5 will become 0 and 1.5 will become 2!
        # If you want to round up in all those cases, do
        # splits += 0.5 instead of round() before casting to int
        splits_indices = splits.round().astype(np.int)
#        if splits_indices[-1] >= len(fragment):
#            splits_indices[-1] -= 1  
        
        split_fragment = np.split(fragment, splits_indices)
        
        quality_vector = []
        
        quality_vector.extend([Q_Score.TERRIBLE.value]*len(split_fragment[0]))
        quality_vector.extend([Q_Score.LOW.value]*len(split_fragment[1]))
        quality_vector.extend([Q_Score.MEDIUM.value]*len(split_fragment[2]))
        quality_vector.extend([Q_Score.HIGH.value]*len(split_fragment[3]))
        quality_vector.extend([Q_Score.MEDIUM.value]*len(split_fragment[4]))
        quality_vector.extend([Q_Score.LOW.value]*len(split_fragment[5]))
        quality_vector.extend([Q_Score.TERRIBLE.value]*len(split_fragment[6]))
        
        assert len(quality_vector) == len(fragment)
        
        return quality_vector        


class Population():
    
    def __init__(self, num_strains = 1000, NUM_MARKERS = 1, marker_length = 1000, num_snps = 3):
        
        
        self.markers = []
        self.strains = []
        self.strain_abundances = []

        for i in range(NUM_MARKERS):
            
            m = Marker(marker_length, num_snps)
            self.markers.append(m)
        
        
        for i in range(num_strains):
            
            # mutate = True because even though each strain has the same set of markers, the 
            # we want the bases of a marker to different in a couple positions between strains.
            copied_markers = copy.deepcopy(self.markers)
            new_strain = Strain(copied_markers, mutate = True) 
            self.strains.append(new_strain)
    
    
    def __str__(self):
        return_str = "Population\n"
        for n, i in enumerate(self.strains):
            return_str += "===========================\n Strain " + str(n+1) + " out of " + str(len(self.strains)) + "\n" + str(i) + "\n"
        return return_str
    
    
class Strain():
    
    def __init__(self, markers, mutate = False):
        self.markers = markers


        if mutate:
            self.mutate_markers()

    def mutate_markers(self):
        """
        Mutate each of this strain's markers.
        """
        
        for marker in self.markers:
            marker.mutate()
                    
    def __str__(self):
        return_str = ""
        for n, i in enumerate(self.markers):
            return_str += "---------------------------\n Marker " + str(n+1) + " out of " + str(len(self.markers)) + "\n" + str(i) + "\n"
        return_str += "\n"
        return return_str
                
class Marker():
    
    def __init__(self, marker_length, num_snps):
        
        self.sequence = [random.choice(["A", "G", "C", "U"]) for i in range(marker_length)]
        
        # Choose evenly spaced SNP locations. 
        # syntax ref: https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
        self.snp_locations = np.round(np.linspace(0, marker_length - 1, num_snps)).astype(int) 
        self.snp_values = [self.sequence[i] for i in self.snp_locations]
        
        
    def mutate(self):
        """
        For each SNP location in this marker, 
        randomly choose a nucleotide to replace the current
        nucleotideat that SNP.
        """
        
        for idx in self.snp_locations:
            current_letter =  self.sequence[idx]
            remaining_letters = [i for i in ["A", "G", "C", "U"] if i !=  current_letter]
            self.sequence[idx] = random.choice(remaining_letters)
            
        self.snp_values = [self.sequence[i] for i in self.snp_locations]
            
        
    def __str__(self):
        return "Sequence: " + str(self.sequence) + "\nSNP Locations: " + str(self.snp_locations) + "\nSNP Values: " + str(self.snp_values)
    

class Model():
    
    # -----------------------------
    # Define class/static variables
    
    Q_DISTRIBUTION = Q_Score_Distribution()

    # Define base change probability matrices conditioned on quality score level.
    
    # Example:
    # HIGH_Q_BASE_CHANGE_MATRIX[_A][_U] is the probability of observing U when the actual
    # nucleotide is _A
    
    _A = 0
    _U = 1
    _C = 2
    _G = 3
    
            
    TERRIBLE_Q_BASE_CHANGE_MATRIX = np.array(([0.25, 0.25, 0.25, 0.25],
                                               [0.25, 0.25, 0.25, 0.25],
                                               [0.25, 0.25, 0.25, 0.25],
                                               [0.25, 0.25, 0.25, 0.25]))
    
    LOW_Q_BASE_CHANGE_MATRIX = np.array(([0.70, 0.10, 0.10, 0.10],
                                          [0.10, 0.70, 0.10, 0.10],
                                          [0.10, 0.10, 0.70, 0.10],
                                          [0.10, 0.10, 0.10, 0.70]))
    
    MEDIUM_Q_BASE_CHANGE_MATRIX = np.array(([0.85, 0.05, 0.05, 0.05],
                                         [0.05, 0.85, 0.05, 0.05],
                                         [0.05, 0.05, 0.85, 0.05],
                                         [0.05, 0.05, 0.05, 0.85]))

    HIGH_Q_BASE_CHANGE_MATRIX = np.array(([0.91, 0.03, 0.03, 0.03],
                                       [0.03, 0.91, 0.03, 0.03],
                                       [0.03, 0.03, 0.91, 0.03],
                                       [0.03, 0.03, 0.03, 0.91]))
        
    Q_SCORE_BASE_CHANGE_MATRICES = [TERRIBLE_Q_BASE_CHANGE_MATRIX,
                                    LOW_Q_BASE_CHANGE_MATRIX,
                                    MEDIUM_Q_BASE_CHANGE_MATRIX,
                                    HIGH_Q_BASE_CHANGE_MATRIX]
    
    
    def __init__(self, fragment_length = 8, num_fragments = 6, num_time_steps = 10, num_strains = 4, num_markers = 1):
        
        
        self.num_time_steps = num_time_steps

        self.fragment_length = fragment_length
        self.num_fragments = num_fragments
        
        self.num_strains = num_strains
        self.marker_length = self.fragment_length * 30000
        self.num_snps = self.marker_length/100 # One SNP every 100 nucleotides 
    
        self.driver = np.array([0]*self.num_strains) # Intialize gaussian process. 
        
        self.bacteria_pop = Population(num_strains=self.num_strains, 
                                       NUM_MARKERS=num_markers, 
                                       marker_length=self.marker_length,
                                       num_snps = self.num_snps)
    
    def run(self):
        
        for i in range(self.num_time_steps):
            print('===========================')
            print("Time step #", i+1, "\n")
            self.step()
    
    def step(self): 
        
        # Generate fragments for this time step.
        fragments = ["".join([random.choice(["A", "G", "C", "U"]) for i in range(self.fragment_length)]) for i in range(self.num_fragments)]
        
        # Step 0
        strain_fragment_matrix = self.generate_strain_fragment_frequencies(fragments, self.bacteria_pop.strains)
        print("Strain Fragment Frequency matrix (each column is one strain, each row is a fragment): \n", strain_fragment_matrix, "\n")
    
        # assert column totals sum to 1 or all entries in column are 0
        for col, col_sum in enumerate(strain_fragment_matrix.sum(axis=0)):
            assert col_sum.round() == 1 or all([not strain_fragment_matrix[x][col] for x in range(len(strain_fragment_matrix))])
        
        # Step 1
        self.driver = self.generate_brownian_motion(self.driver)
        print("Gaussian process: ", self.driver, "\n")
        
        # Step 2
        rel_abundances = self.generate_relative_abundances(self.driver)
        print("Relative abundances of strains: ", rel_abundances, "\n")
        
        self.bacteria_pop.strain_abundances = rel_abundances
        
        # Step 3
        time_indexed_fragment_frequencies = self.generate_time_indexed_fragment_frequencies(strain_fragment_matrix, self.bacteria_pop.strain_abundances)
    
        print("Time indexed Fragment Frequencies")
        print(time_indexed_fragment_frequencies, "\n")
        
        # Step 4
        noisy_reads = self.generate_noisy_reads(time_indexed_fragment_frequencies, fragments)
        print("Sample Noisy Reads")
        print(noisy_reads, "\n")

    # Step 0: 
    def generate_strain_fragment_frequencies(self, fragments, strains):
        """
        @param - fragments --
            A list of fragments
            
        @param - strains --
            A list of strain objects, where each strain has a list of marker sequences.
            
        @returns --
            a 2D numpy array where column i is the relative frequencies of observing each
            of the fragments in strain i.
        """
        
        def count_substring(substring, string):
            """
            Returns the number of times substring occurs in string.
            """
            
            count = 0
            for i in range(len(string)-(len(substring))+1):
                if string[i:i+len(substring)] == substring:
                    count += 1
            return count
    
        def count_fragment_in_strain(fragment, strain):
            """
            Helper function for generate_strain_fragment_frequencies
            
            @fragment - A string of "A" "U" "G" and "C"s.
            
            @strain -- a bacteria strain object. Contains a markers field
                which is a list of lists representing a list of markers, where each marker
                is represented as a list of characters ("A", "U", "G", "C")
            
            @returns --
                the number of times  fragment "fragment" is observed in strain "strain"'s markers
            """      
            
            total = 0
            for marker in strain.markers:
                sequence_string = "".join(marker.sequence)
        
                total += count_substring(fragment, sequence_string)
            
            return total
    
        w = np.zeros((len(fragments), len(strains)))
        
        for col, strain in enumerate(strains):
    
            for row, fragment in enumerate(fragments):
                w[row][col] = count_fragment_in_strain(fragment, strain)
            
            
             # normalize along columns
            column_total = sum([w[row][col] for row in range(len(fragments))])
            if round(column_total): # If column total is not zero, divide each entry by column total
                for row in range(len(fragments)):
                    w[row][col] = w[row][col]/column_total
            
        return w
    
    # Step 1
    def generate_brownian_motion(self, means_vec, time_constant = 1):
        """
        Returns a multivariate gaussian centered according to means_vec
        
        @param - means_vec -- 
            a numpy array of doubles 
            
        @return --
            a numpy array of the same length as the input, chosen from a multiplevariate 
            normal distribution centered at means_vec.
        """
        
        covar_matrix = np.identity(means_vec.size)
        return np.random.multivariate_normal(means_vec, covar_matrix*1/(time_constant**2))
    
    # Step 2
    def generate_relative_abundances(self, gaussian_process):
        """
        @param - gaussain_process - an n-dimensional array of floats
        
        @return - rel_abundances
            a list with same length as guassian_process after running softmax on it.
        """
        
        # softmax
        total = sum(np.exp(gaussian_process))
        rel_abundances = [np.exp(x)/total for x in gaussian_process] 
        
        return rel_abundances
    
    # Step 3
    def generate_time_indexed_fragment_frequencies(self, strain_fragment_matrix, strain_abundances):
        """
        @param - strain_fragment_matrix - 
            A 2D numpy array where column i is the relative frequencies of observing each
            of the fragments in strain i.
            
        @param - strain_abundances
            A list representing the relative abundances of the strains
            
        @returns - fragment_to_prob_vector
             a 1D numpy array of time indexed fragment abundances based on the current time index strain abundances
             and the fragments' relative frequencies in each strain's sequence.
        """
        
        fragment_to_prob_vector = np.matmul(strain_fragment_matrix, np.array(strain_abundances))
        return fragment_to_prob_vector
    
    # Step 4
    def generate_noisy_reads(self, time_indexed_fragment_frequencies, fragments):
        """
        Given a set of fragments and their time indexed frequencies (based on the current time
        index strain abundances and the fragments' relative frequencies in each strain's sequence.), 
        generate a set of noisy fragment reads where the read fragments are selected in proportion
        to their time indexed frequencies and the outputted base pair at location i of each selected 
        fragment is chosen from a probability distribution condition on the actual base pair at location
        i and the quality score at location i in the generated quality score vector for the 
        selected fragment.
        
        @param - time_indexed_fragment_frequencies - 
                a list of floats representing a probability distribution over the fragments
                
        @param - fragments
                a list of strings representing the fragments
        
        @return - generated_noisy_fragments
                a list of strings representing a noisy reads of the set of input fragments
      
        """

    
        # Sample fragments in proportion to their time indexed frequencies.
        sampled_fragments = np.random.choice(fragments, len(fragments), p=time_indexed_fragment_frequencies)
        
        generated_noisy_fragments = []
        
        # For each sampled fragment, generate a noisy read of it.
        for sampled_fragment in sampled_fragments:
            
            # Generate quality score vector from the sample fragment
            quality_score_vector = Model.Q_DISTRIBUTION.get_simple_q_score_vector(list(sampled_fragment))
            
            generated_noisy_fragment = ""
            
            # Generate base pair reads from the sample fragment, conditioned on actual base pair and quality score for that base pair.
            for actual_base_pair, q_score in zip(sampled_fragment, quality_score_vector):
    
                actual_base_pair_index = eval(str("Model._" + actual_base_pair)) # Gets row index in the base pair change matrices.
                
                # Generate a noisy base pair read from the distribution defined by the actual base pair and the quality score
                noisy_letter = np.random.choice(["A", "U", "C", "G"], 1, p = Model.Q_SCORE_BASE_CHANGE_MATRICES[q_score][actual_base_pair_index])[0]
                generated_noisy_fragment  += noisy_letter
            
            generated_noisy_fragments.append(generated_noisy_fragment)
            
        return generated_noisy_fragments


# %%

if __name__ == "__main__":
    random.seed(123)

    my_model = Model()
    my_model.run()
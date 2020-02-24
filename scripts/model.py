#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:41:50 2020

@author: Daniel Alfonsetti
"""


import numpy as np
import random 
import copy
from enum import Enum
import itertools


random.seed(123)



class Population():
    
    def __init__(self, NUM_STRAINS = 1000, NUM_MARKERS = 1, MARKER_LENGTH = 1000, NUM_SNPS = 3):
        
        
        self.markers = []
        self.strains = []
        self.abundances = []

        for i in range(NUM_MARKERS):
            
            m = Marker(MARKER_LENGTH, NUM_SNPS)
            self.markers.append(m)
        
        
        for i in range(NUM_STRAINS):
            
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
        
        for marker in self.markers:
            marker.mutate()
                    
    def __str__(self):
        return_str = ""
        for n, i in enumerate(self.markers):
            return_str += "---------------------------\n Marker " + str(n+1) + " out of " + str(len(self.markers)) + "\n" + str(i) + "\n"
        return_str += "\n"
        return return_str
                
class Marker():
    
    def __init__(self, MARKER_LENGTH, NUM_SNPS):
        
        self.sequence = [random.choice(["A", "G", "C", "U"]) for i in range(MARKER_LENGTH)]
        
        # Choose evenly spaced SNP locations. 
        # syntax ref: https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
        self.snp_locations = np.round(np.linspace(0, MARKER_LENGTH - 1, NUM_SNPS)).astype(int) 
        self.snp_values = [self.sequence[i] for i in self.snp_locations]
        
        
    def mutate(self):
        
        for idx in self.snp_locations:
            current_letter =  self.sequence[idx]
            remaining_letters = [i for i in ["A", "G", "C", "U"] if i !=  current_letter]
            self.sequence[idx] = random.choice(remaining_letters)
            
        self.snp_values = [self.sequence[i] for i in self.snp_locations]
            
        
    def __str__(self):
        return "Sequence: " + str(self.sequence) + "\nSNP Locations: " + str(self.snp_locations) + "\nSNP Values: " + str(self.snp_values)
    



def generate_brownian_motion(means_vec, time_constant = 1):
    """
    Returns a multivariate gaussian centered according to means_vec
    
    Arguments
    means_vec -- an s dimensional array
    """
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.multivariate_normal.html
    
    covar_matrix = np.identity(means_vec.size)
    return np.random.multivariate_normal(means_vec, covar_matrix*1/(time_constant**2))


def generate_relative_abundances(gaussian_process):
    """
    Returns an array with same length as guassian_process after running softmax on it.
    
    Arguments
    gaussain_process - an n-dimensional array
    """
    
    # softmax
    total = sum(np.exp(gaussian_process))
    rel_abundances = [np.exp(x)/total for x in gaussian_process] 
    
    return rel_abundances



def get_fragment_in_strain_prob(fragment, strain):
    """
    Helper function for generate_fragment_frequency
    
    Returns the probability of observing fragment "fragment" in strain "strain"
    """
    
    # TODO: Implement. Must use sliding window scheme.
    

    def count_substring(substring, string):

        count = 0
        for i in range(len(string)-(len(substring))+1):
            if string[i:i+len(substring)] == substring:
                count += 1
        return count
    
    
    total = 0
    for marker in strain.markers:
        sequence_string = "".join(marker.sequence)

        total += count_substring(fragment, sequence_string)
    
    return total


def generate_strain_fragment_frequencies(fragments, bacteria_pop):
    """
        Returns a numpy matrix 
    """
    # TODO: Make this a method of population
    w = np.zeros((len(fragments), len(bacteria_pop.strains)))
    
    for col, strain in enumerate(bacteria_pop.strains):

        for row, fragment in enumerate(fragments):
            w[row][col] = get_fragment_in_strain_prob(fragment, strain)
        
        
         # normalize along columns
        column_total = sum([w[row][col] for row in range(len(fragments))])
        if round(column_total): # If column total is not zero, divide each entry by column total
            for row in range(len(fragments)):
                w[row][col] = w[row][col]/column_total
        
    return w
            
    
    
def generate_time_indexed_fragment_frequencies(strain_fragment_matrix, bacteria_pop):
    """
    
    returns a dictionary mapping fragments to their fraction
    """
    
    fragment_to_prob_vector = np.matmul(strain_fragment_matrix, np.array(bacteria_pop.abundances))
    return fragment_to_prob_vector



class Q_Score(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    TERRIBLE = 0


class Q_Score_Distribution():
    
    def __init__(self):
        
        self.distribution = {Q_Score.HIGH : 0.5, Q_Score.MEDIUM : 0.3, Q_Score.LOW : 0.15, Q_Score.TERRIBLE: 0.05}
    
    def get_q_score_vector(self, fragment):
        """
            fragment is a list
            Modified from: 
                https://stackoverflow.com/questions/32330459/partition-a-list-into-sublists-by-percentage
        """
        
        
        percentages = [self.distribution[Q_Score.TERRIBLE]/2, 
                  self.distribution[Q_Score.LOW]/2, 
                  self.distribution[Q_Score.MEDIUM]/2, 
                  self.distribution[Q_Score.HIGH], 
                  self.distribution[Q_Score.MEDIUM]/2, 
                  self.distribution[Q_Score.LOW]/2, 
                  self.distribution[Q_Score.TERRIBLE]/2]
        
        
        splits = np.cumsum(percentages)
        print(splits)

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
        
        
        print(splits_indices)
        split_fragment = np.split(fragment, splits_indices)
        
        quality_vector = []
        
        quality_vector.extend([Q_Score.TERRIBLE]*len(split_fragment[0]))
        quality_vector.extend([Q_Score.LOW]*len(split_fragment[1]))
        quality_vector.extend([Q_Score.MEDIUM]*len(split_fragment[2]))
        quality_vector.extend([Q_Score.HIGH]*len(split_fragment[3]))
        quality_vector.extend([Q_Score.MEDIUM]*len(split_fragment[4]))
        quality_vector.extend([Q_Score.LOW]*len(split_fragment[5]))
        quality_vector.extend([Q_Score.TERRIBLE]*len(split_fragment[6]))
        
        assert len(quality_vector) == len(fragment)
        
        return quality_vector        
  
      


# Take from https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-52.php
def all_repeat(str1, rno):
  chars = list(str1)
  results = []
  for c in itertools.product(chars, repeat = rno):
    results.append(''.join(c))
  return results

# %% #

if __name__ == "__main__":

    random.seed(123)
    NUM_TIME_STEPS = 10
    NUMBER_STRAINS = 4
    
 
    # FRAGMENTS = all_repeat("AGCU", 2) # size 16. This would come from real world data.
    # FRAGMENTS
    
    FRAGMENT_LENGTH = 9
    FRAGMENT_NUM = 8
    FRAGMENTS = ["".join([random.choice(["A", "G", "C", "U"]) for i in range(FRAGMENT_LENGTH)]) for i in range(FRAGMENT_NUM)]

    driver = np.array([0]*NUMBER_STRAINS) # Intialize gaussian process. 
    
    bacteria_pop = Population(NUM_STRAINS=NUMBER_STRAINS, NUM_MARKERS=1, MARKER_LENGTH=FRAGMENT_LENGTH*10000, NUM_SNPS = 5000)
    print(bacteria_pop)

    strain_fragment_matrix = generate_strain_fragment_frequencies(FRAGMENTS, bacteria_pop)
    print(strain_fragment_matrix)

    # assert column totals sum to 1 or all entries in column are 0
    for col, col_sum in enumerate(strain_fragment_matrix.sum(axis=0)):
        assert col_sum.round() == 1 or all([not strain_fragment_matrix[x][col] for x in range(len(strain_fragment_matrix))])
        
    
    #%% 
    
    for i in range(NUM_TIME_STEPS):
        print('===========================')
        print("Step #", i, "\n")
              
        driver = generate_brownian_motion(driver)
        print("Gaussian process: ", driver, "\n")
        
        rel_abundances = generate_relative_abundances(driver)
        print("Relative abundances of strains: ", rel_abundances, "\n")
        
        bacteria_pop.abundances = rel_abundances
        
        fragments_to_abundance = generate_time_indexed_fragment_frequencies(strain_fragment_matrix, bacteria_pop)
    
        print("Fragments to abundance")
        print(fragments_to_abundance, "\n")
    
#    my_pop = Population(NUM_STRAINS = 3, NUM_MARKERS = 2, MARKER_LENGTH = 10, NUM_SNPS = 5 )
#    print(my_pop)

# %%
        
    
q_distribution = Q_Score_Distribution()
      
        
q_distribution.get_q_score_vector(list(FRAGMENTS[0]))
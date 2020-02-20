#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:41:50 2020

@author: Daniel Alfonsetti
"""


import numpy as np
import random 
import copy


random.seed(123)


class Marker():
    
    def __init__(self, MARKER_LENGTH, NUM_SNPS):
        
        self.sequence = [random.choice(["A", "G", "C", "U"]) for i in range(MARKER_LENGTH)]
        
        # Choose evenly space SNP locations. 
        # syntax ref: https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
        self.snp_locations = np.round(np.linspace(0, MARKER_LENGTH - 1, NUM_SNPS)).astype(int) 
        
    def __str__(self):
        return "Sequence: " + str(self.sequence) + "\nSNP Locations: " + str(self.snp_locations) 
    
#    def __repr__(self):
#        return "Sequence: " + str(self.sequence) + "\n " + str(self.snp_locations)


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
            for marker in self.markers:
                for idx in marker.snp_locations:
                    current_letter =  marker.sequence[idx]
                    remaining_letters = [i for i in ["A", "G", "C", "U"] if i !=  current_letter]
                    marker.sequence[idx] = random.choice(remaining_letters)
                    
    def __str__(self):
        return_str = ""
        for n, i in enumerate(self.markers):
            return_str += "---------------------------\n Marker " + str(n+1) + " out of " + str(len(self.markers)) + "\n" + str(i) + "\n"
        return_str += "\n"
        return return_str
                
    


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


    
def generate_fragment_frequency(fragments, bacteria_pop):
    """
    
    returns a dictionary mapping fragments to their fraction
    """
    
    
    def fragment_in_strain_prob(fragment, strain_markers):
        """
        Helper function for generate_fragment_frequency
        
        Returns the probability of observing fragment "fragment" in strain "strain"
        """
        
        # TODO: Implement 
        
        # print(strain_markers)
        return 0.05
    
    
    fragment_to_prob = {}
    for fragment in fragments:
        print("===========")

        total_prob = 0
        
        for strain_prob, strain in zip(bacteria_pop.abundances, bacteria_pop.strains):
            print("-----------")
            fragment_prob = fragment_in_strain_prob(fragment, strain.markers)
            
            total_prob += strain_prob*fragment_prob
            print(strain_prob)
            print(fragment_prob)
            print(total_prob)
        
        fragment_to_prob[fragment] = total_prob
        
    return fragment_to_prob
    
    

from itertools import product

# Take from https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-52.php
def all_repeat(str1, rno):
  chars = list(str1)
  results = []
  for c in product(chars, repeat = rno):
    results.append(''.join(c))
  return results


if __name__ == "__main__":

    TIME_STEPS = 10
    NUMBER_STRAINS = 4
    FRAGMENTS = all_repeat("AGCU", 2) # size 1024

    driver = np.array([0]*NUMBER_STRAINS) # Intialize gaussian process. 
    
    
    bacteria_pop = Population(NUM_STRAINS=NUMBER_STRAINS, NUM_MARKERS=1, MARKER_LENGTH=20, NUM_SNPS = 3)
    
    for i in range(0, TIME_STEPS):
        print("Step #", i)
              
        driver = generate_brownian_motion(driver)
        print("Gaussian process: ", driver)
        
        rel_abundances = generate_relative_abundances(driver)
        print("Relative abundances of strains: ", rel_abundances)
        
        bacteria_pop.abundances = rel_abundances
        
        fragments_to_abundance = generate_fragment_frequency(FRAGMENTS, bacteria_pop)
    
        print("Fragments to abundance")
        print(fragments_to_abundance)
    
    
#    my_pop = Population(NUM_STRAINS = 3, NUM_MARKERS = 2, MARKER_LENGTH = 10, NUM_SNPS = 5 )
#    print(my_pop)
    
        
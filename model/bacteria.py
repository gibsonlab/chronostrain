import copy
import random
import numpy as np


class Population:

    def __init__(self, num_strains=1000, num_markers=1, marker_length=1000, num_snps=3):

        self.num_strains = num_strains
        self.markers = []
        self.strains = []
        self.strain_abundances = []

        for i in range(num_markers):
            m = Marker(marker_length, num_snps)
            self.markers.append(m)

        for i in range(num_strains):
            # mutate = True because even though each strain has the same set of markers, the
            # we want the bases of a marker to different in a couple positions between strains.
            copied_markers = copy.deepcopy(self.markers)
            new_strain = Strain(copied_markers, mutate=True)
            self.strains.append(new_strain)

    def get_fragment_space(self, window_size):
        """
            Returns a list of strings, where the strings are  all distinct fragments of
            size "window_size" among the markers of all strains in this population.
        """

        fragment_space = set()
        for strain in self.strains:
            for marker in strain.markers:
                for i in range(len(marker.sequence) - window_size + 1):
                    fragment_str = "".join(marker.sequence[i:i + window_size])
                    if fragment_str not in fragment_space:
                        fragment_space.add(fragment_str)

        return list(fragment_space)

    def generate_strain_fragment_frequencies(self, fragment_space):
        """

        @param - a list of fragment strings.

        @returns --
            a 2D numpy array where column i is the relative frequencies of observing each
            of the fragments in strain i.
        """

        def count_substring(substring, string):
            """
            Returns the number of times substring occurs in string.
            """

            count = 0
            for i in range(len(string) - (len(substring)) + 1):
                if string[i:i + len(substring)] == substring:
                    count += 1
            return count

        def count_fragment_in_strain(fragment, strain):
            """
            Helper function for generate_strain_fragment_frequencies

            @fragment - A string of "A" "T" "G" and "C"s.

            @strain -- a bacteria strain object. Contains a markers field
                which is a list of lists representing a list of markers, where each marker
                is represented as a list of characters ("A", "T", "G", "C")

            @returns --
                the number of times  fragment "fragment" is observed in strain "strain"'s markers
            """

            total = 0
            for marker in strain.markers:
                sequence_string = "".join(marker.sequence)

                total += count_substring(fragment, sequence_string)

            return total

        W = np.zeros((len(fragment_space), len(self.strains)))  # (rows, columns)

        for col, strain in enumerate(self.strains):

            for row, fragment in enumerate(fragment_space):
                W[row][col] = count_fragment_in_strain(fragment, strain)

            # normalize along columns
            column_total = sum([W[row][col] for row in range(len(fragment_space))])
            if round(column_total):  # If column total is not zero, divide each entry by column total
                for row in range(len(fragment_space)):
                    W[row][col] = W[row][col] / column_total

        if not all([round(sum(W[:, i]), 4) == 1 for i in range(W.shape[1])]):
            raise ValueError("Expected all columns in W matrix to sum to 1")

        return W

    def __str__(self):
        return_str = "Population\n"
        for n, i in enumerate(self.strains):
            return_str += "===========================\n Strain " + str(n + 1) + " out of " + str(
                len(self.strains)) + "\n" + str(i) + "\n"
        return return_str


class Strain:

    def __init__(self, markers, mutate=False):
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
            return_str += "---------------------------\n Marker " + str(n + 1) + " out of " + str(
                len(self.markers)) + "\n" + str(i) + "\n"
        return_str += "\n"
        return return_str


class Marker:

    def __init__(self, marker_length, num_snps):
        self.sequence = [random.choice(["A", "G", "C", "T"]) for i in range(marker_length)]

        # Choose evenly spaced SNP locations.
        # syntax ref:
        # https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last

        self.snp_locations = np.round(np.linspace(0, marker_length - 1, num_snps)).astype(int)
        self.snp_values = [self.sequence[i] for i in self.snp_locations]

    def mutate(self):
        """
        For each SNP location in this marker,
        randomly choose a nucleotide to replace the current
        nucleotideat that SNP.
        """

        for idx in self.snp_locations:
            current_letter = self.sequence[idx]
            remaining_letters = [i for i in ["A", "G", "C", "T"] if i != current_letter]
            self.sequence[idx] = random.choice(remaining_letters)

        self.snp_values = [self.sequence[i] for i in self.snp_locations]

    def __str__(self):
        return "Sequence: " + str(self.sequence) + "\nSNP Locations: " + str(
            self.snp_locations) + "\nSNP Values: " + str(self.snp_values)


class RealStrains:
    # TODO: Implement real strains with real markers
    def __init__(self, species):

        self.markers = None





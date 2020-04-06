import numpy as np


class Population:
    def __init__(self, strains):
        """
        :param strains: a list of Strain instances.
        """

        if not all([isinstance(s, Strain) for s in strains]):
            raise ValueError("All elements in strains must be Strain instances")

        self.strains = strains  # A list of Strain objects.
        self.fragment_space_map = {}  # Maps window sizes (ints) to their corresponding fragment space (list of strings)
        self.fragment_frequencies_map = {}  # Maps window sizes to their corresponding fragment frequencies matrices.

    def get_fragment_space(self, window_size):
        """
            Get all fragments of a particular window size; lazy instantiation.
            Returns a list of strings, where the strings are all distinct fragments of
            size "window_size" among the markers of all strains in this population.
        """
        if window_size in self.fragment_space_map.keys():
            return self.fragment_space_map[window_size]

        fragment_space = set()
        for strain in self.strains:
            for marker in strain.markers:
                for i in range(len(marker.nucleotides) - window_size + 1):
                    fragment_str = "".join(marker.nucleotides[i:i + window_size])
                    if fragment_str not in fragment_space:
                        fragment_space.add(fragment_str)

        fragment_space = list(fragment_space)
        self.fragment_space_map[window_size] = fragment_space

        return fragment_space

    def generate_strain_fragment_frequencies(self, window_size):
        """

        Get fragment counts per strain

        @param - a list of fragment strings.

        @returns --
            a 2D numpy array where column i is the relative frequencies of observing each
            of the fragments in strain i.
        """

        def count_substring(substring, string):
            """
            Helper function.
            Returns the number of times substring occurs in string.
            """

            count = 0
            for i in range(len(string) - (len(substring)) + 1):
                if string[i:i + len(substring)] == substring:
                    count += 1
            return count

        def count_fragment_in_strain(fragment, strain):
            """
            Helper function.

            @fragment - A string of "A" "T" "G" and "C"s.

            @strain -- a bacteria strain object. Contains a markers field
                which is a list of lists representing a list of markers, where each marker
                is represented as a list of characters ("A", "T", "G", "C")

            @returns --
                the number of times  fragment "fragment" is observed in strain "strain"'s markers
            """

            total = 0
            for marker in strain.markers:
                sequence_string = "".join(marker.nucleotides)

                total += count_substring(fragment, sequence_string)

            return total
        #################

        if window_size in self.fragment_frequencies_map.keys():

            return self.fragment_frequencies_map[window_size]

        fragment_space = self.get_fragment_space(window_size)

        # fragment_frequencies represents W matrix
        fragment_frequencies = np.zeros((len(fragment_space), len(self.strains)))

        # For each strain, fill in a column.
        for col, strain in enumerate(self.strains):

            # For each row (fragment) in the current column (strain), insert that fragment's relative frequency
            # in that strain compared to the other fragments in the fragment space.
            for row, fragment in enumerate(fragment_space):
                fragment_frequencies[row][col] = count_fragment_in_strain(fragment, strain)

            # normalize along columns
            column_total = sum([fragment_frequencies[row][col] for row in range(len(fragment_space))])
            if round(column_total):  # If column total is not zero, divide each entry by column total
                for row in range(len(fragment_space)):
                    fragment_frequencies[row][col] = fragment_frequencies[row][col] / column_total

        if not all([round(sum(fragment_frequencies[:, i]), 4) == 1 for i in range(fragment_frequencies.shape[1])]):
            raise ValueError("Expected all columns in W matrix to sum to 1")

        self.fragment_frequencies_map[window_size] = fragment_frequencies
        return fragment_frequencies


class Strain:
    def __init__(self, markers, name=""):
        """
        :param markers: a list of Marker instances.
        :param name: Strain name or accession number.
        """

        if not all([isinstance(s, Marker) for s in markers]):
            raise ValueError("All elements in markers must be Marker instances")

        self.name = name
        self.markers = markers

    def __str__(self):
        return_str = ""
        for n, i in enumerate(self.markers):
            return_str += "---------------------------\n Marker " + str(n + 1) + " out of " + str(
                len(self.markers)) + "\n" + str(i) + "\n"
        return_str += "\n"
        return return_str


class Marker:
    def __init__(self, nucleotides):
        """
        :param nucleotides: a string of A, G, T, and C characters.
        """

        if not isinstance(nucleotides, str):
            raise ValueError("nucleotides must be a string")

        self.nucleotides = nucleotides

    def __str__(self):
        return "Sequence: " + str(self.nucleotides)






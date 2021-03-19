import torch
from dataclasses import dataclass
from typing import List, Union

from chronostrain.config import cfg
from chronostrain.model.fragments import FragmentSpace
from chronostrain.util.io.logger import logger


@dataclass
class MarkerMetadata:
    strain_accession: str
    subseq_name: str
    file_path: str
    
    def __repr__(self):
        return "{}-{}".format(self.strain_accession, self.subseq_name)
        
    def __str__(self):
        return self.__repr__()


@dataclass
class StrainMetadata:
    ncbi_accession: str
    name: str
    file_path: str


@dataclass
class Marker:
    name: str
    seq: str
    metadata: Union[MarkerMetadata, None]

    def __repr__(self):
        return "Marker[{}:{}]".format(self.name, self.seq) if MarkerMetadata is None else "Marker[{}:{}]".format(self.metadata, self.seq)

    def __str__(self):
        return self.__repr__()


@dataclass
class Strain:
    name: str
    markers: List[Marker]
    genome_length: int
    metadata: StrainMetadata

    def __repr__(self):
        return "Strain({})".format(self.name)

    def __str__(self):
        return self.__repr__()


class Population:
    def __init__(self, strains: List[Strain]):
        """
        :param strains: a list of Strain instances.
        """

        if not all([isinstance(s, Strain) for s in strains]):
            raise ValueError("All elements in strains must be Strain instances")

        self.strains = strains  # A list of Strain objects.
        self.fragment_space_map = {}  # Maps window sizes (ints) to their corresponding fragment space (list of strings)
        self.fragment_frequencies_map = {}  # Maps window sizes to their corresponding fragment frequencies matrices.

    def get_fragment_space(self, window_size) -> FragmentSpace:
        """
            Retrieves the fragment space via lazy instantiation.
            Returns a FragmentSpace instance.
        """
        if window_size in self.fragment_space_map.keys():
            return self.fragment_space_map[window_size]

        logger.debug("Constructing fragment space for window size {}...".format(window_size))
        fragment_space = FragmentSpace()
        for strain in self.strains:
            for marker in strain.markers:
                for seq, pos in sliding_window(marker.seq, window_size):
                    fragment_space.add_seq(seq, metadata=strain.name + "Pos" + str(pos))

        self.fragment_space_map[window_size] = fragment_space
        logger.debug("Finished constructing fragment space.")

        return fragment_space

    def get_strain_fragment_frequencies(self, window_size) -> torch.Tensor:
        """
        Get fragment counts per strain. The output represents the 'W' matrix in the notes.
        :param window_size: an integer specifying the fragment window length.
        :return: An (F x S) matrix, where each column is a strain-specific frequency vector of fragments.
        """

        # Lazy initialization; Return if already initialized.
        if window_size in self.fragment_frequencies_map.keys():
            return self.fragment_frequencies_map[window_size]

        # For each strain, fill out the column.
        fragment_space = self.get_fragment_space(window_size)
        frag_freqs = torch.zeros(fragment_space.size(), len(self.strains), device=cfg.torch_cfg.device)

        logger.debug("Constructing fragment frequencies for window size {}...".format(window_size))

        for col, strain in enumerate(self.strains):
            for marker in strain.markers:
                for subseq, _ in sliding_window(marker.seq, window_size):
                    frag_freqs[fragment_space.get_fragment(subseq).index][col] += 1

        # normalize each col to sum to 1.
        frag_freqs = frag_freqs / torch.tensor([
            [strain.genome_length - window_size + 1 for strain in self.strains]
        ], device=cfg.torch_cfg.device)

        self.fragment_frequencies_map[window_size] = frag_freqs
        logger.debug("Finished constructing fragment frequencies for window size {}.".format(window_size))
        return frag_freqs


def sliding_window(seq, width):
    """
    A generator for the subsequences produced by a sliding window of specified width.
    """
    for i in range(len(seq) - width + 1):
        yield seq[i:i + width], i

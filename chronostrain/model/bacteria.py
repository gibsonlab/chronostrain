import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.config import cfg
from chronostrain.model.fragments import FragmentSpace
from chronostrain.util.sparse import normalize_sparse_2d
from . import logger


@dataclass
class MarkerMetadata:
    parent_accession: str
    gene_id: str
    file_path: Path
    
    def __repr__(self):
        return self.gene_id
        
    def __str__(self):
        return self.__repr__()


@dataclass
class StrainMetadata:
    ncbi_accession: str
    file_path: Path
    genus: str
    species: str


@dataclass
class Marker:
    name: str
    seq: str
    metadata: Union[MarkerMetadata, None]

    def __repr__(self):
        if self.metadata is None:
            return "Marker[{}:{}]".format(self.name, self.seq)
        else:
            return "Marker[{}({}):{}]".format(self.name, self.metadata, self.seq)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.seq)

    def to_seqrecord(self, description: str = "") -> SeqRecord:
        return SeqRecord(
            Seq(self.seq),
            id="{}|{}|{}".format(self.metadata.parent_accession, self.name, self.metadata.gene_id),
            description=description
        )


@dataclass
class Strain:
    id: str  # Typically, ID is the accession number.
    markers: List[Marker]
    genome_length: int
    metadata: StrainMetadata

    def __repr__(self):
        return "Strain({})".format(self.id)

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
                    fragment_space.add_seq(seq, metadata="{}_{}_Pos({})".format(strain.id, marker.metadata.gene_id, pos))

        self.fragment_space_map[window_size] = fragment_space
        logger.debug("Finished constructing fragment space. (Size={})".format(fragment_space.size()))

        return fragment_space

    def get_strain_fragment_frequencies(self, window_size) -> torch.Tensor:
        """
        Get fragment counts per strain. The output represents the 'W' matrix in the notes.
        :param window_size: an integer specifying the fragment window length.
        :return: An (F x S) matrix, where each column is a strain-specific frequency vector of fragments.
        """
        # Simplification: Assume uniform read length per fragment. This is not a theoretically necessary assumption,
        #   but a practically necessary one.

        # Return if already created.
        if window_size in self.fragment_frequencies_map.keys():
            return self.fragment_frequencies_map[window_size]

        # Couldn't find existing instance. Create a new one.
        if cfg.model_cfg.use_sparse:
            frag_freqs = self.get_strain_fragment_frequencies_sparse(window_size)
        else:
            frag_freqs = self.get_strain_fragment_frequencies_dense(window_size)
        self.fragment_frequencies_map[window_size] = frag_freqs
        logger.debug("Finished constructing fragment frequencies for window size {}.".format(window_size))
        return frag_freqs

    def get_strain_fragment_frequencies_dense(self, window_size) -> torch.Tensor:
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
        return frag_freqs

    def get_strain_fragment_frequencies_sparse(self, window_size) -> torch.Tensor:
        # For each strain, fill out the column.
        fragment_space = self.get_fragment_space(window_size)
        logger.debug("Constructing fragment frequencies for window size {}...".format(window_size))

        strain_indices = []
        frag_indices = []
        matrix_values = []

        for strain_idx, strain in enumerate(self.strains):
            for marker in strain.markers:
                for subseq, _ in sliding_window(marker.seq, window_size):
                    strain_indices.append(strain_idx)
                    frag_indices.append(fragment_space.get_fragment(subseq).index)
                    matrix_values.append(1)

        # normalize each col to sum to 1.
        return normalize_sparse_2d(
            torch.tensor([frag_indices, strain_indices], device=cfg.torch_cfg.device, dtype=torch.long),
            torch.tensor(matrix_values, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype),
            fragment_space.size(),
            len(self.strains),
            0
        ).coalesce()


def sliding_window(seq, width):
    """
    A generator for the subsequences produced by a sliding window of specified width.
    """
    for i in range(len(seq) - width + 1):
        yield seq[i:i + width], i

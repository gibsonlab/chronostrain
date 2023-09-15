from typing import List, Iterator
from .strain import Strain
from .marker import Marker

class Population:
    def __init__(self, strains: List[Strain]):
        """
        :param strains: a list of Strain instances.
        """

        if not all([isinstance(s, Strain) for s in strains]):
            raise ValueError("All elements in strains must be Strain instances")

        self.strains = strains  # A list of Strain objects.
        self.strain_indices = {strain: s_idx for s_idx, strain in enumerate(self.strains)}

    def __hash__(self):
        """
        Returns a hashed representation of the strain collection.
        :return:
        """
        return "[{}]".format(
            ",".join(strain.__repr__() for strain in self.strains)
        ).__hash__()

    def __repr__(self):
        return self.strains.__repr__()

    def __str__(self):
        return "[{}]".format(
            ",".join(strain.__str__() for strain in self.strains)
        )

    def num_strains(self) -> int:
        return len(self.strains)

    def __len__(self) -> int:
        return self.num_strains()

    def markers_iterator(self) -> Iterator[Marker]:
        for strain in self.strains:
            for marker in strain.markers:
                yield marker

    def strain_index(self, strain: Strain) -> int:
        """
        Return the relative index of the strain (useful for resolving matrix index positions)
        """
        return self.strain_indices[strain]
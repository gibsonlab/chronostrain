from pathlib import Path
from typing import List, Iterator, Union
from .strain import Strain
from .marker import Marker
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.util.external import samtools


class StrainCollection:
    def __init__(self, strains: List[Strain], db_signature: str):
        """
        :param strains: a list of Strain instances.
        :param db_signature: a compact string description/hashed signature of the database that these strains came from.
        """

        if not all([isinstance(s, Strain) for s in strains]):
            raise ValueError("All elements must be Strain instances")

        self.strains = strains  # A list of Strain objects.
        self.strain_indices = {strain: s_idx for s_idx, strain in enumerate(self.strains)}
        self.db_signature = db_signature
        self._multifasta_path: Union[Path, None] = None  # Lazy initialization

    @property
    def multifasta_file(self) -> Path:
        if self._multifasta_path is None:
            from chronostrain.config import cfg
            from Bio import SeqIO
            self.cache = ComputationCache(
                CacheTag(
                    cache_dir=cfg.model_cfg.cache_dir,
                    strains=[s.id for s in self.strains],
                    database=self.db_signature
                )
            )
            self._multifasta_path = self.cache.create_subdir('db_index').resolve() / 'markers.fasta'

            # Create the index if it does not already exist.
            if not self._multifasta_path.exists():
                tmp_path = self._multifasta_path.with_suffix('.fasta.TEMP')
                with open(tmp_path, "w"):
                    records = []
                    for strain in self.strains:
                        for marker in strain.markers:
                            records.append(marker.to_seqrecord(description=""))
                    SeqIO.write(records, tmp_path, "fasta")
                tmp_path.rename(self._multifasta_path)
        return self._multifasta_path

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

    def __iter__(self) -> Iterator[Strain]:
        yield from self.strains

    def markers_iterator(self) -> Iterator[Marker]:
        for strain in self.strains:
            for marker in strain.markers:
                yield marker

    def strain_index(self, strain: Strain) -> int:
        """
        Return the relative index of the strain (useful for resolving matrix index positions)
        """
        return self.strain_indices[strain]

    @property
    def faidx_file(self) -> Path:
        tgt_path = self.multifasta_file.parent / f'{self.multifasta_file.name}.fai'
        if not tgt_path.exists():
            samtools.faidx(self.multifasta_file)
        return tgt_path

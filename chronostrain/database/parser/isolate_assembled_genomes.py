from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List
import pandas as pd

from .base import AbstractDatabaseParser
from chronostrain.model import Strain, Marker, MarkerMetadata, StrainMetadata
from chronostrain.util.io import read_seq_file

from .. import StrainDatabase
from ..backend import PandasAssistedBackend
from ...util.sequences import FastaIndexedResource, DynamicFastaSequence

from chronostrain.logging import create_logger
logger = create_logger(__name__)


@dataclass
class AssemblySpecification(object):
    accession: str
    fasta_path: Path
    genus: str
    species: str


def parse_specification(spec_path: Path) -> Iterator[AssemblySpecification]:
    """
    Parse a TSV-formatted file with the columns:
    [Accession,FastaPath,Genus,Species]
    """
    spec_df = pd.read_csv(spec_path, sep='\t')
    if spec_df.shape[0] == 0:
        raise ValueError("Specified database metadata file is empty.")
    if not {'Accession', 'FastaPath', 'Genus', 'Species'}.issubset(set(spec_df.columns)):
        raise ValueError("Expected at least 4 columns [Accession,FastaPath,Genus,Species]. Verify the metadata file.")

    for _, row in spec_df.iterrows():
        yield AssemblySpecification(
            row['Accession'],
            Path(row['FastaPath']),
            row['Genus'],
            row['Species']
        )


def parse_strain_from_assembly(spec: AssemblySpecification) -> Strain:
    """
    Parses the scaffolds/contigs into a Strain object. Uses the entire genome (or as much of it that was assembled).
    """
    logger.debug("Parsing assembly {} ({} {}) from resource {}".format(
        spec.accession,
        spec.genus,
        spec.species,
        spec.fasta_path
    ))
    is_gzipped = spec.fasta_path.suffix.endswith('.gz')
    seq_resource = FastaIndexedResource(spec.fasta_path, is_gzipped)

    markers = []
    total_bases = 0
    for i, contig_record in enumerate(read_seq_file(spec.fasta_path, "fasta")):
        markers.append(
            Marker(
                id=contig_record.id.replace('|', ':'),
                name=f'Contig-{i}:{spec.accession}',
                seq=DynamicFastaSequence(seq_resource, contig_record.id),
                metadata=MarkerMetadata(spec.accession, str(spec.fasta_path))
            )
        )
        total_bases += len(contig_record.seq)

    logger.debug("{} Contigs found. [{:,} bases]".format(len(markers), total_bases))
    return Strain(
        id=spec.accession,
        name=f'{spec.genus}{spec.species}_{spec.accession}',
        markers=markers,
        metadata=StrainMetadata(
            genus=spec.genus,
            species=spec.species,
            total_len=-1,
            cluster=[]
        )
    )


class IsolateAssemblyParser(AbstractDatabaseParser):
    def __init__(self, db_name: str, specs: Path, data_dir: Path):
        super().__init__(
            db_name=db_name,
            data_dir=data_dir
        )
        self.specifications: List[AssemblySpecification] = list(parse_specification(specs))

    def strains(self) -> Iterator[Strain]:
        for spec in self.specifications:
            yield parse_strain_from_assembly(spec)

    def parse(self) -> StrainDatabase:
        try:
            db = self.load_from_disk()
            logger.debug("Loaded database instance from {}.".format(self.disk_path()))
            return db
        except FileNotFoundError:
            pass

        backend = PandasAssistedBackend()
        backend.add_strains(self.strains())
        db = StrainDatabase(
            backend=backend,
            name=self.db_name,
            data_dir=self.data_dir
        )
        self.save_to_disk(db)
        return db

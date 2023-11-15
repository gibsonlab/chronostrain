from pathlib import Path
from typing import List

from chronostrain.database import StrainDatabase
from chronostrain.model import Strain
from chronostrain.util.cache import ComputationCache, CacheTag
from chronostrain.config import cfg
from chronostrain.model.io import TimeSeriesReads


class ReadStrainCollectionCache(ComputationCache):
    """
    A subclass of ComputationCache representing the results of likelihood matrix computation
     associated with a particular TimeSeriesReads instance and collection of Strains (variants).

    The specific cache dependency is:
    1) Configuration's use_quality_scores flag,
    2) The bacterial strain database,
    3) The read file paths, in order of specification (if the order changes, then so does the cache key.)
    """
    def __init__(self, reads: TimeSeriesReads, db: StrainDatabase, strains: List[Strain]):
        super().__init__(
            CacheTag(
                use_quality=cfg.model_cfg.use_quality_scores,
                database=db.signature,
                strains=[s.id for s in strains],
                file_paths=[
                    src_path
                    for reads_t in reads
                    for src_path in reads_t.paths()
                ]  # read files
            )
        )
        self.strains = strains
        self.marker_fasta_path = self.create_subdir('db_index').resolve() / 'markers.fasta'
        self.init_marker_multifasta()

    @property
    def faidx_file(self) -> Path:
        return self.marker_fasta_path.with_suffix('.fai')

    def init_marker_multifasta(self):
        from Bio import SeqIO
        if self.marker_fasta_path.exists():
            return

        tmp_path = self.marker_fasta_path.with_suffix('.fasta.TEMP')
        with open(tmp_path, "w"):
            records = []
            for strain in self.strains:
                for marker in strain.markers:
                    records.append(marker.to_seqrecord(description=""))
            SeqIO.write(records, tmp_path, "fasta")
        tmp_path.rename(self.marker_fasta_path)

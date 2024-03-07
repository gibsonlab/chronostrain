import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any, Union

from chronostrain.model import Strain, StrainMetadata, Marker

from .base import AbstractDatabaseParser, StrainDatabaseParseError
from .marker_sources import MultiFastaMarkerSource, AbstractMarkerSource
from .. import StrainDatabase
from ...util.sequences import UnknownNucleotideError

from chronostrain.logging import create_logger
logger = create_logger(__name__)


# =====================================================================
# JSON entry dataclasses. Each class implements a deserialize() method.
# =====================================================================

class StrainKeyMissingError(StrainDatabaseParseError):
    pass


def extract_key_from_json(json_obj: dict, key: str):
    try:
        return json_obj[key]
    except KeyError:
        raise StrainKeyMissingError(f"Missing entry `{key}` from json entry {json_obj}.")


def extract_optional_key_from_json(json_obj: dict, key: str, default: Any):
    return json_obj.get(key, default)


@dataclass
class StrainEntry:
    id: str
    genus: str
    species: str
    genome_length: int
    strain_name: str
    seq_entries: List["SeqEntry"]
    marker_entries: List["MarkerEntry"]
    cluster: List[str]

    def __str__(self):
        return "(Strain Entry: {genus} {species})".format(
            genus=self.genus,
            species=self.species
        )

    def __repr__(self):
        return "{}(id={})".format(
            self.__class__.__name__,
            self.id
        )

    def marker_entries_by_seq(self) -> Iterator[Tuple['SeqEntry', List['MarkerEntry']]]:
        # Group markers by their source acessions.
        grouping: Dict[str, List[MarkerEntry]] = defaultdict(list)
        for marker_entry in self.marker_entries:
            grouping[marker_entry.source_seq].append(marker_entry)

        # Iterate through source accessions.
        src_accessions_expected = set(grouping.keys())
        for seq_entry in self.seq_entries:
            if seq_entry.seq_id in src_accessions_expected:
                src_accessions_expected.remove(seq_entry.seq_id)
            _marker_entries = grouping[seq_entry.seq_id]
            if len(_marker_entries) > 0:
                yield seq_entry, _marker_entries

        if len(src_accessions_expected) > 0:
            raise StrainDatabaseParseError(
                "Markers of strain `{}` requested the sources [{}], which were not specified.".format(
                    self.id,
                    ",".join(src_accessions_expected)
                )
            )

    @staticmethod
    def deserialize(json_dict: dict):
        strain_id = extract_key_from_json(json_dict, 'id')
        genus = extract_key_from_json(json_dict, 'genus')
        species = extract_key_from_json(json_dict, 'species')
        strain_name = extract_key_from_json(json_dict, 'name')
        seqs_json = extract_key_from_json(json_dict, 'seqs')
        cluster_json = extract_optional_key_from_json(json_dict, 'cluster', [])
        markers_json = extract_key_from_json(json_dict, 'markers')
        genome_length = int(extract_key_from_json(json_dict, 'genome_length'))

        marker_entries = []
        seq_entries = []
        entry = StrainEntry(id=strain_id,
                            genus=genus,
                            species=species,
                            genome_length=genome_length,
                            strain_name=strain_name,
                            seq_entries=seq_entries,
                            marker_entries=marker_entries,
                            cluster=cluster_json)
        for idx, marker_json_obj in enumerate(markers_json):
            marker_entries.append(MarkerEntry.deserialize(marker_json_obj))
        for idx, seq_json_obj in enumerate(seqs_json):
            seq_entries.append(SeqEntry.deserialize(seq_json_obj))
        return entry


@dataclass
class SeqEntry:
    seq_id: str
    seq_path: Union[Path, None]

    @staticmethod
    def deserialize(entry_dict: dict) -> 'SeqEntry':
        seq_id = extract_key_from_json(entry_dict, 'id')
        try:
            seq_path = Path(extract_key_from_json(entry_dict, 'seq_path'))
        except StrainKeyMissingError:
            seq_path = None
        return SeqEntry(seq_id, seq_path)

    def __repr__(self) -> str:
        return f"SeqEntry[ID={self.seq_id}|path={self.seq_path}]"


class MarkerEntry:
    def __init__(
            self,
            marker_id: str,
            name: str,
            source_seq: str,
            record_idx: int
    ):
        self.marker_id = marker_id
        self.name = name
        self.source_seq = source_seq
        self.record_idx = record_idx

    @staticmethod
    def deserialize(entry_dict: dict) -> "MarkerEntry":
        marker_type = extract_key_from_json(entry_dict, 'type')
        if marker_type == 'subseq':
            return SubseqMarkerEntry.deserialize(entry_dict)  # uses a subsequence of the target fasta entry.
        elif marker_type == 'fasta':
            return FastaRecordEntry.deserialize(entry_dict)  # uses the entire fasta record
        else:
            raise StrainDatabaseParseError("Unexpected type `{}` in marker entry {}".format(
                marker_type, entry_dict
            ))


class SubseqMarkerEntry(MarkerEntry):
    def __init__(
            self,
            marker_id: str, name: str, source_seq: str, record_idx: int,
            start: int, end: int, is_negative_strand: bool
    ):
        super().__init__(marker_id, name, source_seq, record_idx)
        self.start_pos = start
        self.end_pos = end
        self.is_negative_strand = is_negative_strand

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "SubSeq(seq={}${},start={},end={})".format(
            self.source_seq,
            self.record_idx,
            self.start_pos,
            self.end_pos
        )

    @staticmethod
    def deserialize(entry_dict: dict) -> "SubseqMarkerEntry":
        start_pos = int(extract_key_from_json(entry_dict, 'start'))
        end_pos = int(extract_key_from_json(entry_dict, 'end'))
        strand_str = extract_key_from_json(entry_dict, 'strand')
        if strand_str == '+':
            is_negative_strand = False
        elif strand_str == '-':
            is_negative_strand = True
        else:
            raise ValueError(
                f"Unrecognizable value `{strand_str}` of entry `strand` in {entry_dict}."
            )

        return SubseqMarkerEntry(
            marker_id=extract_key_from_json(entry_dict, 'id'),
            name=extract_key_from_json(entry_dict, 'name'),
            source_seq=extract_key_from_json(entry_dict, 'source'),
            record_idx=int(extract_key_from_json(entry_dict, 'source_i')),
            start=start_pos,
            end=end_pos,
            is_negative_strand=is_negative_strand
        )


class FastaRecordEntry(MarkerEntry):
    def __init__(self, marker_id: str, name: str, source_seq: str, record_idx: int):
        super().__init__(marker_id, name, source_seq, record_idx)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"FastaRecord[{self.source_seq}${self.record_idx}]"

    @staticmethod
    def deserialize(entry_dict: dict) -> "FastaRecordEntry":
        return FastaRecordEntry(
            marker_id=extract_key_from_json(entry_dict, 'id'),
            name=extract_key_from_json(entry_dict, 'name'),
            source_seq=extract_key_from_json(entry_dict, 'source'),
            record_idx=int(extract_key_from_json(entry_dict, 'source_i')),
        )


class UnknownSourceNucleotideError(BaseException):
    def __init__(self, e: UnknownNucleotideError, src: str, marker_entry: MarkerEntry):
        self.nucleotide = e.nucleotide
        self.src = src
        self.marker_entry = marker_entry


class JSONParser(AbstractDatabaseParser):
    def __init__(self,
                 data_dir: Union[str, Path],
                 entries_file: Union[str, Path],
                 marker_max_len: int,
                 force_refresh: bool = False):
        if isinstance(entries_file, str):
            entries_file = Path(entries_file)
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        super().__init__(
            db_name=entries_file.stem,
            data_dir=data_dir
        )
        self.entries_file = entries_file
        self.marker_max_len = marker_max_len
        self.force_refresh = force_refresh

    def strain_entries(self) -> Iterator[StrainEntry]:
        """
        Deserialize JSON into StrainEntry instances.
        """
        with open(self.entries_file, "r") as f:
            for idx, strain_json_obj in enumerate(json.load(f)):
                yield StrainEntry.deserialize(strain_json_obj)

    def parse_strain(self, strain_entry: StrainEntry) -> Strain:
        strain_markers = []
        for seq_entry, marker_entries in strain_entry.marker_entries_by_seq():
            # The only exception is when the seq entry is a multi-fasta file.
            marker_src = MultiFastaMarkerSource(
                fasta_path=seq_entry.seq_path,
                strain_id=strain_entry.id,
                seq_id=seq_entry.seq_id
            )

            """
            Note 1: this bit of code is now defunct; all marker sources are expected to load from a local file.
            Note 0: could use CachedEntrezMarkerSource, but this is not necessary since we now use pickle-based caching.
            """
            # marker_src = EntrezMarkerSource(
            #     strain_id=strain_entry.id,
            #     data_dir=self.data_dir,
            #     seq_accession=seq_entry.accession,
            #     seq_path=seq_entry.seq_path,
            #     marker_max_len=self.marker_max_len,
            #     force_download=self.force_refresh
            # )

            for marker_entry in marker_entries:
                try:
                    marker = self.parse_marker(marker_entry, marker_src)
                except UnknownNucleotideError as e:
                    raise UnknownSourceNucleotideError(
                        e,
                        f'{marker_src.seq_id},IDX={marker_entry.record_idx}',
                        marker_entry
                    ) from None

                strain_markers.append(marker)
        if len(strain_markers) == 0:
            logger.warning("No markers parsed for strain entry {}.".format(
                str(strain_entry)
            ))
        else:
            logger.debug("Strain {} ({} {}, {}) loaded with {} markers.".format(
                strain_entry.id,
                strain_entry.genus,
                strain_entry.species,
                strain_entry.strain_name,
                len(strain_markers)
            ))
        return Strain(
            id=strain_entry.id,
            name=strain_entry.strain_name,
            markers=strain_markers,
            metadata=StrainMetadata(
                genus=strain_entry.genus,
                species=strain_entry.species,
                total_len=strain_entry.genome_length,
                cluster=strain_entry.cluster
            )
        )

    def parse_marker(self, marker_entry: MarkerEntry, marker_src: AbstractMarkerSource) -> Marker:
        if isinstance(marker_entry, SubseqMarkerEntry):
            marker = marker_src.extract_subseq(
                marker_entry.record_idx,
                marker_entry.marker_id,
                marker_entry.name,
                marker_entry.start_pos,
                marker_entry.end_pos,
                marker_entry.is_negative_strand
            )
        elif isinstance(marker_entry, FastaRecordEntry):
            # noinspection PyUnresolvedReferences
            # see the implementation of extract_fasta_record for a to-do
            marker = marker_src.extract_fasta_record(
                marker_entry.marker_id,
                marker_entry.name,
                marker_entry.record_id,
                allocate=True
            )
        else:
            raise NotImplementedError(
                f"Unsupported entry class `{marker_entry.__class__.__name__}`."
            )
        return marker

    def strains(self) -> Iterator[Strain]:
        logger.info("Loading from JSON marker database file {}.".format(self.entries_file))
        for strain_entry in self.strain_entries():
            try:
                yield self.parse_strain(strain_entry)
            except UnknownSourceNucleotideError as e:
                logger.warning(
                    f"Encountered an unknown nucleotide {e.nucleotide} (possible IUPAC code) in {e.src}, "
                    f"while parsing marker {e.marker_entry.marker_id}. "
                    f"Skipping strain {strain_entry.id}."
                )
                continue

    def parse(self) -> StrainDatabase:
        try:
            db = self.load_from_disk()
            logger.debug("Loaded database instance from {}.".format(self.disk_path()))
        except FileNotFoundError:
            logger.debug("Couldn't find instance ({}).".format(self.disk_path()))

            from ..backend import DictionaryBackend
            backend = DictionaryBackend()
            backend.add_strains(self.strains())
            db = StrainDatabase(
                backend=backend,
                name=self.db_name,
                data_dir=self.data_dir
            )
            self.save_to_disk(db)

        logger.debug("Using `{}` for database backend.".format(db.backend.__class__.__name__))
        return db

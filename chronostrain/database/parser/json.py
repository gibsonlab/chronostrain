import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple, Dict

from chronostrain.config import cfg
from chronostrain.model import Strain, StrainMetadata, Marker
from chronostrain.config.logging import create_logger

from .base import AbstractDatabaseParser, StrainDatabaseParseError
from .marker_sources import CachedMarkerSource, MarkerSource
from ...util.sequences import UnknownNucleotideError

logger = create_logger(__name__)


# =====================================================================
# JSON entry dataclasses. Each class implements a deserialize() method.
# =====================================================================


def extract_key_from_json(json_obj: dict, key: str):
    try:
        return json_obj[key]
    except KeyError:
        raise StrainDatabaseParseError(f"Missing entry `{key}` from json entry {json_obj}.")


@dataclass
class StrainEntry:
    id: str
    genus: str
    species: str
    strain_name: str
    seq_entries: List["SeqEntry"]
    marker_entries: List["MarkerEntry"]

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
        grouping: Dict[str, List[MarkerEntry]] = defaultdict(list)
        for marker_entry in self.marker_entries:
            grouping[marker_entry.source_accession].append(marker_entry)

        src_accessions_left = set(grouping.keys())
        for seq_entry in self.seq_entries:
            src_accessions_left.remove(seq_entry.accession)
            _marker_entries = grouping[seq_entry.accession]
            if len(_marker_entries) > 0:
                yield seq_entry, _marker_entries

        if len(src_accessions_left) > 0:
            raise StrainDatabaseParseError(
                "Markers of strain `{}` requested the sources [{}], which were not specified.".format(
                    self.id,
                    ",".join(src_accessions_left)
                )
            )

    @staticmethod
    def deserialize(json_dict: dict):
        strain_id = extract_key_from_json(json_dict, 'id')
        genus = extract_key_from_json(json_dict, 'genus')
        species = extract_key_from_json(json_dict, 'species')
        strain_name = extract_key_from_json(json_dict, 'name')
        seqs_json = extract_key_from_json(json_dict, 'seqs')
        markers_json = extract_key_from_json(json_dict, 'markers')

        marker_entries = []
        seq_entries = []
        entry = StrainEntry(id=strain_id,
                            genus=genus,
                            species=species,
                            strain_name=strain_name,
                            seq_entries=seq_entries,
                            marker_entries=marker_entries)
        for idx, marker_json_obj in enumerate(markers_json):
            marker_entries.append(MarkerEntry.deserialize(marker_json_obj))
        for idx, seq_json_obj in enumerate(seqs_json):
            seq_entries.append(SeqEntry.deserialize(seq_json_obj))
        return entry


@dataclass
class SeqEntry:
    accession: str
    seq_type: str  # typically "chromosome" or "scaffold"

    @staticmethod
    def deserialize(entry_dict: dict) -> 'SeqEntry':
        accession = extract_key_from_json(entry_dict, 'accession')
        seq_type = extract_key_from_json(entry_dict, 'seq_type')
        return SeqEntry(accession, seq_type)

    @property
    def is_chromosome(self) -> bool:
        return self.seq_type == "chromosome"

    @property
    def is_scaffold(self) -> bool:
        return self.seq_type == "scaffold"

    @property
    def is_contig(self) -> bool:
        return self.seq_type == "contig"


class MarkerEntry:
    def __init__(self, marker_id: str, name: str, is_canonical: bool, source_accession: str):
        self.marker_id = marker_id
        self.name = name
        self.is_canonical = is_canonical
        self.source_accession = source_accession

    @staticmethod
    def deserialize(entry_dict: dict) -> "MarkerEntry":
        marker_type = extract_key_from_json(entry_dict, 'type')

        if marker_type == 'tag':
            return TagMarkerEntry.deserialize(entry_dict)
        elif marker_type == 'primer':
            return PrimerMarkerEntry.deserialize(entry_dict)
        elif marker_type == 'subseq':
            return SubseqMarkerEntry.deserialize(entry_dict)
        else:
            raise StrainDatabaseParseError("Unexpected type `{}` in marker entry {}".format(
                marker_type, entry_dict
            ))


class TagMarkerEntry(MarkerEntry):
    def __init__(self, marker_id: str, name: str, is_canonical: bool, source_accession: str, locus_tag: str):
        super().__init__(marker_id, name, is_canonical, source_accession)
        self.locus_tag = locus_tag

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "TagMarker[{}:locus={},{}]".format(
            self.source_accession,
            self.locus_tag,
            ":canonical" if self.is_canonical else ""
        )

    @staticmethod
    def deserialize(entry_dict: dict) -> "TagMarkerEntry":
        try:
            is_canonical = str(extract_key_from_json(entry_dict, 'canonical')).strip().lower() == "true"
        except StrainDatabaseParseError:
            is_canonical = False

        return TagMarkerEntry(
            marker_id=extract_key_from_json(entry_dict, 'id'),
            name=extract_key_from_json(entry_dict, 'name'),
            is_canonical=is_canonical,
            source_accession=extract_key_from_json(entry_dict, 'source'),
            locus_tag=extract_key_from_json(entry_dict, 'locus_tag')
        )


class PrimerMarkerEntry(MarkerEntry):
    def __init__(self,
                 marker_id: str, name: str, is_canonical: bool, source_accession: str,
                 forward: str, reverse: str):
        super().__init__(marker_id, name, is_canonical, source_accession)
        self.forward = forward
        self.reverse = reverse

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "PrimerMarker(parent={},fwd={},rev={},Canonical={})".format(
            self.source_accession,
            self.forward,
            self.reverse,
            self.is_canonical
        )

    def entry_id(self) -> str:
        return "{}[Primer:{}]".format(
            self.source_accession,
            '-'.join([self.forward, self.reverse])
        )

    @staticmethod
    def deserialize(entry_dict: dict) -> "PrimerMarkerEntry":
        return PrimerMarkerEntry(
            marker_id=extract_key_from_json(entry_dict, 'id'),
            name=extract_key_from_json(entry_dict, 'name'),
            is_canonical=str(extract_key_from_json(entry_dict, 'canonical')).strip().lower() == "true",
            source_accession=extract_key_from_json(entry_dict, 'source'),
            forward=extract_key_from_json(entry_dict, 'forward'),
            reverse=extract_key_from_json(entry_dict, 'reverse')
        )


class SubseqMarkerEntry(MarkerEntry):
    def __init__(self, marker_id: str, name: str, is_canonical: bool, source_accession: str,
                 start: int, end: int, is_negative_strand: bool):
        super().__init__(marker_id, name, is_canonical, source_accession)
        self.start_pos = start
        self.end_pos = end
        self.is_negative_strand = is_negative_strand

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "SubSeq(parent={},start={},end={},Canonical={})".format(
            self.source_accession,
            self.start_pos,
            self.end_pos,
            self.is_canonical
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
            is_canonical=str(extract_key_from_json(entry_dict, 'canonical')).strip().lower() == "true",
            source_accession=extract_key_from_json(entry_dict, 'source'),
            start=start_pos,
            end=end_pos,
            is_negative_strand=is_negative_strand
        )


class UnknownSourceNucleotideError(BaseException):
    def __init__(self, e: UnknownNucleotideError, src: str):
        self.nucleotide = e.nucleotide
        self.src = src


class JSONParser(AbstractDatabaseParser):
    def __init__(self,
                 entries_file: Path,
                 marker_max_len: int,
                 force_refresh: bool = False):
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
        chromosome_accs = []
        scaffold_accs = []
        contig_accs = []
        total_len = 0
        for seq_entry, marker_entries in strain_entry.marker_entries_by_seq():
            if seq_entry.is_chromosome:
                chromosome_accs.append(seq_entry.accession)
            elif seq_entry.is_scaffold:
                scaffold_accs.append(seq_entry.accession)
            elif seq_entry.is_contig:
                contig_accs.append(seq_entry.accession)

            marker_src = CachedMarkerSource(
                strain_id=strain_entry.id,
                seq_accession=seq_entry.accession,
                marker_max_len=self.marker_max_len,
                force_download=self.force_refresh
            )

            try:
                for marker in self.parse_markers(marker_entries, marker_src):
                    strain_markers.append(marker)
            except UnknownNucleotideError as e:
                raise UnknownSourceNucleotideError(e, marker_src.seq_accession) from None

            total_len += marker_src.nucleotide_length
        if len(strain_markers) == 0:
            logger.warning("No markers parsed for strain entry {}.".format(
                str(strain_entry)
            ))
        else:
            logger.debug("Strain {} loaded with {} markers.".format(
                strain_entry.id,
                len(strain_markers)
            ))

        return Strain(
            id=strain_entry.id,
            name=strain_entry.strain_name,
            markers=strain_markers,
            metadata=StrainMetadata(
                chromosomes=chromosome_accs,
                scaffolds=scaffold_accs + contig_accs,  # Treat these as the same in the metadata.
                genus=strain_entry.genus,
                species=strain_entry.species,
                total_len=total_len
            )
        )

    def parse_markers(self, marker_entries: List[MarkerEntry], marker_src: MarkerSource) -> Iterator[Marker]:
        for marker_entry in marker_entries:
            if isinstance(marker_entry, TagMarkerEntry):
                marker = marker_src.extract_from_locus_tag(
                    marker_entry.marker_id,
                    marker_entry.name,
                    marker_entry.is_canonical,
                    marker_entry.locus_tag
                )
            elif isinstance(marker_entry, PrimerMarkerEntry):
                marker = marker_src.extract_from_primer(
                    marker_entry.marker_id,
                    marker_entry.name,
                    marker_entry.is_canonical,
                    marker_entry.forward,
                    marker_entry.reverse
                )
            elif isinstance(marker_entry, SubseqMarkerEntry):
                marker = marker_src.extract_subseq(
                    marker_entry.marker_id,
                    marker_entry.name,
                    marker_entry.is_canonical,
                    marker_entry.start_pos,
                    marker_entry.end_pos,
                    marker_entry.is_negative_strand
                )
            else:
                raise NotImplementedError(
                    f"Unsupported entry class `{marker_entry.__class__.__name__}`."
                )
            yield marker

    def strains(self) -> Iterator[Strain]:
        logger.info("Loading from JSON marker database file {}.".format(self.entries_file))
        logger.debug("Database root directory: {}".format(cfg.database_cfg.data_dir))
        for strain_entry in self.strain_entries():
            try:
                yield self.parse_strain(strain_entry)
            except UnknownSourceNucleotideError as e:
                logger.warning(f"Skipping strain entry {strain_entry.id}, "
                               f"due to unknown nucleotide {e.nucleotide} (possible IUPAC code) in {e.src}.")
                continue

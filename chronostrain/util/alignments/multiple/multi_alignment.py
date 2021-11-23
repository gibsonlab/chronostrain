from pathlib import Path
from typing import Dict, Tuple, Iterator, List

import Bio.AlignIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

from chronostrain.database import StrainDatabase
from chronostrain.model import Marker, SequenceRead
from chronostrain.model.io import TimeSeriesReads
from ...external import clustal_omega, mafft_fragment, mafft_global
from ...sequences import *

from chronostrain.config import create_logger
logger = create_logger(__name__)


_READ_PREFIX = "READ"
_SEPARATOR = "#"


class MarkerMultipleFragmentAlignment(object):
    """
    Encapsulates a multiple alignment of a marker sequence and short reads.
    """
    def __init__(self,
                 marker_idxs: Dict[Marker, int],
                 aligned_marker_seqs: SeqType,
                 read_multi_alignment: SeqType,
                 forward_read_index_map: Dict[SequenceRead, int],
                 reverse_read_index_map: Dict[SequenceRead, int],
                 start_clips: List[int],
                 end_clips: List[int],
                 time_idxs: np.ndarray,
                 file_path: Path
                 ):
        if aligned_marker_seqs.shape[1] != read_multi_alignment.shape[1]:
            raise ValueError("Read alignments must be of the same length as the marker alignment string.")
        if read_multi_alignment.shape[0] != len(forward_read_index_map) + len(reverse_read_index_map):
            raise ValueError("Each row of the read multiple alignment must be specified by a read id in either "
                             "the forward or reverse mappings.")
        if aligned_marker_seqs.shape[0] != len(marker_idxs):
            raise ValueError("Each row of the marker multiple alignment must be specified by a marker instance.")

        self.marker_idxs = marker_idxs
        self.aligned_marker_seqs = aligned_marker_seqs
        self.read_multi_alignment = read_multi_alignment
        self.time_idxs = time_idxs
        self.forward_read_index_map = forward_read_index_map
        self.reverse_read_index_map = reverse_read_index_map
        self.start_clips = start_clips
        self.end_clips = end_clips
        self.file_path = file_path
        self.canonical_marker = self.find_canonical_marker()

    def markers(self) -> Iterator[Marker]:
        yield from self.marker_idxs.keys()

    def find_canonical_marker(self):
        for marker in self.markers():
            if marker.is_canonical:
                return marker
        raise RuntimeError("Couldn't find canonical marker in multiple alignment {}".format(
            self.file_path
        ))

    def num_bases(self) -> int:
        return self.aligned_marker_seqs.shape[1]

    def contains_read(self, read: SequenceRead, revcomp: bool) -> bool:
        if revcomp:
            return read in self.reverse_read_index_map
        else:
            return read in self.forward_read_index_map

    def get_index_of_read(self, read: SequenceRead, revcomp: bool) -> int:
        if revcomp:
            return self.reverse_read_index_map[read]
        else:
            return self.forward_read_index_map[read]

    def get_index_of_marker(self, marker: Marker) -> int:
        return self.marker_idxs[marker]

    def get_aligned_read_seq(self, read: SequenceRead, revcomp: bool) -> SeqType:
        read_idx = self.get_index_of_read(read, revcomp)
        return self.read_multi_alignment[read_idx, :]

    def get_aligned_marker_seq(self, marker: Marker) -> SeqType:
        marker_idx = self.get_index_of_marker(marker)
        return self.aligned_marker_seqs[marker_idx]

    def get_alignment(self,
                      marker: Marker,
                      read: SequenceRead,
                      revcomp: bool,
                      delete_double_gaps: bool = True) -> SeqType:
        read_seq = self.get_aligned_read_seq(read, revcomp)
        marker_seq = self.get_aligned_marker_seq(marker)

        if delete_double_gaps:
            return self.delete_double_gaps(marker_seq, read_seq)
        else:
            return np.stack([
                marker_seq, read_seq
            ], axis=0)

    def aln_gapped_boundary(self, read: SequenceRead, revcomp: bool) -> Tuple[int, int]:
        """
        Find the first and last ungapped positions.
        :return: A tuple (i, j) such that aln_seq[i] != GAP and aln_seq[j] != GAP, and all elements to the left of i, and
        to the right of j, are GAPs.
        """
        if revcomp:
            r_idx = self.reverse_read_index_map[read]
        else:
            r_idx = self.forward_read_index_map[read]

        aln_seq = self.read_multi_alignment[r_idx]
        ungapped_indices = np.where(aln_seq != nucleotide_GAP_z4)[0]
        return ungapped_indices[0], ungapped_indices[-1]

    @staticmethod
    def get_boundary_of_aligned_seq(aln_seq: SeqType):
        ungapped_indices = np.where(aln_seq != nucleotide_GAP_z4)[0]
        return ungapped_indices[0],  ungapped_indices[-1]

    def get_aligned_reference_region(self,
                                     marker: Marker,
                                     read: SequenceRead,
                                     revcomp: bool) -> Tuple[SeqType, np.ndarray, np.ndarray, int, int]:
        """
        Returns the aligned fragment (with gaps removed), and a pair of boolean arrays (insertion, deletion).
        The insertion array indicates which positions of the read (with gaps removed) are insertions,
        and the deletion array indicates which positions of the fragment (with gaps removed) are deleted in the read.
        """

        aln = self.get_alignment(marker, read, revcomp, delete_double_gaps=True)
        first, last = self.get_boundary_of_aligned_seq(aln[1])
        aln = aln[:, first:last+1]

        marker_section = aln[0]
        read_section = aln[1]

        insertion_locs = np.equal(marker_section, nucleotide_GAP_z4)
        # Get rid of indices corresponding to deletions.
        insertion_locs = insertion_locs[read_section != nucleotide_GAP_z4]

        deletion_locs = np.equal(read_section, nucleotide_GAP_z4)
        # Get rid of indices corresponding to insertions.
        deletion_locs = deletion_locs[marker_section != nucleotide_GAP_z4]

        start_clip, end_clip = self.num_clipped_bases(read, revcomp)

        return marker_section[marker_section != nucleotide_GAP_z4], insertion_locs, deletion_locs, start_clip, end_clip

    def reads(self, revcomp: bool) -> Iterator[SequenceRead]:
        if revcomp:
            yield from self.reverse_read_index_map.keys()
        else:
            yield from self.forward_read_index_map.keys()

    @staticmethod
    def delete_double_gaps(marker_aln: SeqType, read_aln: SeqType) -> SeqType:
        """
        Eliminate from the pair of alignment strings the indices where both sequences simultaneously have gaps.
        """
        ungapped_indices = (marker_aln != nucleotide_GAP_z4) | (read_aln != nucleotide_GAP_z4)
        return np.stack([
            marker_aln[ungapped_indices], read_aln[ungapped_indices]
        ], axis=0)

    def num_clipped_bases(self, read: SequenceRead, revcomp: bool) -> Tuple[int, int]:
        if revcomp:
            r_idx = self.reverse_read_index_map[read]
        else:
            r_idx = self.forward_read_index_map[read]
        return self.start_clips[r_idx], self.end_clips[r_idx]


def parse(db: StrainDatabase, target_marker_name: str, reads: TimeSeriesReads, aln_path: Path) -> MarkerMultipleFragmentAlignment:
    forward_reads: List[SequenceRead] = []
    forward_seqs: List[SeqType] = []
    forward_time_idxs: List[int] = []
    forward_start_clips: List[int] = []
    forward_end_clips: List[int] = []

    reverse_reads: List[SequenceRead] = []
    reverse_seqs: List[SeqType] = []
    reverse_time_idxs: List[int] = []
    reverse_start_clips: List[int] = []
    reverse_end_clips: List[int] = []

    marker_idxs: Dict[Marker, int] = {}
    marker_seqs: List[SeqType] = []
    marker_regions: List[Tuple[int, int]] = []

    # ============================ BEGIN HELPERS ============================
    def parse_marker_record(marker_record: SeqRecord):
        record_tokens = marker_record.id.split("|")
        parsed_marker_name = record_tokens[1]
        parsed_marker_id = record_tokens[2]
        if target_marker_name != parsed_marker_name:
            raise ValueError(f"Expected marker `{target_marker_name}`, "
                             f"but instead found `{parsed_marker_name}` in alignment file.")
        aligned_marker_seq = nucleotides_to_z4(str(marker_record.seq))

        # The marker sequence's aligned region. Keep track of this to clip off the start/end edge effects.
        matched_indices = np.where(aligned_marker_seq != nucleotide_GAP_z4)[0]
        marker_start = matched_indices[0]
        marker_end = matched_indices[-1]

        marker_seqs.append(aligned_marker_seq)
        marker_regions.append((marker_start, marker_end))
        marker_idxs[db.get_marker(parsed_marker_id)] = len(marker_seqs) - 1

    def parse_read_record(read_record: SeqRecord, start_clip: int, end_clip: int):
        # Parse the tokens in the ID.
        tokens = read_record.id.split(_SEPARATOR)
        t_idx = int(tokens[1])
        rev_comp = int(tokens[2]) == 1
        read_id = tokens[3]

        # Get the read instance.
        read_obj = reads[t_idx].get_read(read_id)

        # Check if alignment is clipped off the edges of the marker.
        aln_seq = nucleotides_to_z4(str(read_record.seq))
        n_start_clip = np.sum(aln_seq[:start_clip] != nucleotide_GAP_z4)
        n_end_clip = np.sum(aln_seq[end_clip + 1:] != nucleotide_GAP_z4)

        aln_seq = aln_seq[start_clip:end_clip + 1]

        # Store the alignment into the proper category (whether or not the read was reverse complemented).
        if not rev_comp:
            forward_reads.append(read_obj)
            forward_seqs.append(aln_seq)
            forward_start_clips.append(n_start_clip)
            forward_end_clips.append(n_end_clip)
            forward_time_idxs.append(t_idx)
        else:
            reverse_reads.append(read_obj)
            reverse_seqs.append(aln_seq)
            reverse_start_clips.append(n_start_clip)
            reverse_end_clips.append(n_end_clip)
            reverse_time_idxs.append(t_idx)
    # ============================ END HELPERS ============================
    # Parse the marker entry first.

    for record in Bio.AlignIO.read(str(aln_path), 'fasta'):
        if not record.id.startswith(f"{_READ_PREFIX}{_SEPARATOR}"):
            parse_marker_record(record)

    if len(marker_idxs) == 0:
        raise RuntimeError("Couldn't find any reference marker in multiple alignment output {}".format(
            aln_path
        ))

    start_clip = min(entry[0] for entry in marker_regions)
    end_clip = max(entry[1] for entry in marker_regions)

    # Parse the other entries.
    for record in Bio.AlignIO.read(str(aln_path), 'fasta'):
        if record.id.startswith(f"{_READ_PREFIX}{_SEPARATOR}"):
            parse_read_record(record, start_clip, end_clip)

    # Build the mappings.
    forward_read_index_map = {}
    reverse_read_index_map = {}
    for idx, read in enumerate(forward_reads):
        if read in forward_read_index_map:
            raise RuntimeError(f"Found repeat entry forward-mapped read {read.id}.")
        forward_read_index_map[read] = idx
    for idx, read in enumerate(reverse_reads):
        if read in reverse_read_index_map:
            raise RuntimeError(f"Found repeat entry for reverse-mapped read {read.id}.")
        reverse_read_index_map[read] = idx + len(forward_reads)

    return MarkerMultipleFragmentAlignment(
        marker_idxs=marker_idxs,
        aligned_marker_seqs=np.stack(
            [marker_seq[start_clip:end_clip + 1] for marker_seq in marker_seqs],
            axis=0
        ),
        read_multi_alignment=np.stack(forward_seqs + reverse_seqs, axis=0),
        start_clips=forward_start_clips + reverse_start_clips,
        end_clips=forward_end_clips + reverse_end_clips,
        forward_read_index_map=forward_read_index_map,
        reverse_read_index_map=reverse_read_index_map,
        time_idxs=np.array(forward_time_idxs + reverse_time_idxs, dtype=int),
        file_path=aln_path
    )


def align(db: StrainDatabase,
          marker_name: str,
          read_descriptions: Iterator[Tuple[int, SequenceRead, bool]],
          intermediate_fasta_path: Path,
          out_fasta_path: Path,
          n_threads: int = 1):
    markers = db.get_markers_by_name(marker_name)
    marker_profile_path = markers[0].metadata.file_path.parent / f"{marker_name}_profile.fasta"
    create_marker_profile(marker_profile_path, markers)

    align_mafft(marker_profile_path, read_descriptions, intermediate_fasta_path, out_fasta_path, n_threads)
    # align_clustalo(marker, read_descriptions, intermediate_fasta_path, out_fasta_path, n_threads)


def create_marker_profile(profile_path: Path, markers: List[Marker], n_threads: int = 1):
    marker_fasta_path = profile_path.parent / f"{profile_path.stem}_input.fasta"

    SeqIO.write(
        [marker.to_seqrecord() for marker in markers],
        marker_fasta_path,
        "fasta"
    )

    mafft_global(
        input_fasta_path=marker_fasta_path,
        output_path=profile_path,
        n_threads=n_threads,
        auto=True,
        max_iterates=1000
    )

    marker_fasta_path.unlink()


def align_mafft(marker_profile_path: Path,
                read_descriptions: Iterator[Tuple[int, SequenceRead, bool]],
                intermediate_fasta_path: Path,
                out_fasta_path: Path,
                n_threads: int = 1):
    """
    Write these records to file (using a predetermined format), then perform multiple alignment.
    """
    # First write to temporary file, with the reads reverse complemented if necessary.
    records = []
    for t_idx, read, should_reverse_comp in read_descriptions:
        if should_reverse_comp:
            read_seq = reverse_complement_seq(read.seq)
        else:
            read_seq = read.seq

        if should_reverse_comp:
            revcomp_flag = 1
        else:
            revcomp_flag = 0

        record = SeqRecord(
            Seq(z4_to_nucleotides(read_seq)),
            id=f"{_READ_PREFIX}{_SEPARATOR}{t_idx}{_SEPARATOR}{revcomp_flag}{_SEPARATOR}{read.id}",
            description=f"(Read, t: {t_idx}, reverse complement:{should_reverse_comp})"
        )
        records.append(record)
    SeqIO.write(records, intermediate_fasta_path, "fasta")

    logger.debug(f"Invoking `mafft --addfragments` on {len(records)} sequences.")

    # Now invoke MAFFT aligner.
    mafft_fragment(
        reference_fasta_path=marker_profile_path,
        fragment_fasta_path=intermediate_fasta_path,
        output_path=out_fasta_path,
        n_threads=n_threads,
        auto=True,
        gap_open_penalty_group=1.53,
        gap_offset_group=0.0,
        jtt_pam=1,
        tm_pam=1
    )


def align_clustalo(marker: Marker,
                   read_descriptions: Iterator[Tuple[int, SequenceRead, bool]],
                   intermediate_fasta_path: Path,
                   out_fasta_path: Path,
                   n_threads: int = 1):
    """
    Write these records to file (using a predetermined format), then perform multiple alignment.

    Experimental 11/22/2021: Forces progressive alignment in the order that the reads come in.
    """
    # First write to temporary file, with the reads reverse complemented if necessary.
    records = []
    record_ids = []

    # record = marker.to_seqrecord()
    # records.append(record)
    # record_ids.append(record.id)

    for t_idx, read, should_reverse_comp in read_descriptions:
        if should_reverse_comp:
            read_seq = reverse_complement_seq(read.seq)
        else:
            read_seq = read.seq

        if should_reverse_comp:
            revcomp_flag = 1
        else:
            revcomp_flag = 0

        read_id = f"{_READ_PREFIX}{_SEPARATOR}{t_idx}{_SEPARATOR}{revcomp_flag}{_SEPARATOR}{read.id}"

        record = SeqRecord(
            Seq(z4_to_nucleotides(read_seq)),
            id=read_id,
            description=f"(Read, t: {t_idx}, reverse complement:{should_reverse_comp})"
        )
        records.append(record)
        record_ids.append(read_id)
    SeqIO.write(records, intermediate_fasta_path, "fasta")

    if len(record_ids) >= 2:
        tree_path = intermediate_fasta_path.parent / "guide_tree"
        create_tree_canonical(record_ids, tree_path)
    else:
        tree_path = None

    logger.debug(
        f"Invoking `clustalo` on {len(records)} sequences. "
        f"Using {marker.metadata.file_path.name} as profile. (May take a while)"
    )

    # Now invoke Clustal-Omega aligner.
    clustal_omega(
        input_path=intermediate_fasta_path,
        output_path=out_fasta_path,
        force_overwrite=True,
        verbose=False,
        out_format='fasta',
        seqtype='DNA',
        n_threads=n_threads,
        guidetree_in=tree_path,
        profile1=marker.metadata.file_path.parent / "profile.fasta"
    )


def create_tree_canonical(ids: List[str], path: Path):
    assert len(ids) >= 2
    tree_str = f"(\n{ids[0]}\n,\n{ids[1]}\n)"

    for x in ids[2:]:
        tree_str = f"(\n{tree_str}\n,\n{x}\n)"

    with open(path, "w") as f:
        print(tree_str, file=f)
        print(";", file=f)

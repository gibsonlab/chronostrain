import enum
import re
from typing import List, Iterable, Dict

from Bio import SeqIO
from chronostrain.util.sequences import complement_seq


class SamTags(enum.Enum):
    ReadName = 0
    MapFlag = 1
    ContigName = 2
    MapPos = 3
    MapQuality = 4
    Cigar = 5
    MatePair = 6
    MatePos = 7
    TemplateLen = 8
    Read = 9
    Quality = 10


class MapFlags(enum.Enum):
    """
    The mapping types given in the MapFlag tag. The actual tag is given bitwise,
    so the presence of these tags is found as:
    (Line[SamTags.MapFlag] & MapFlags.flag == MapFlags.flag)
    """
    ReadIsPaired = 1
    Unmapped = 4
    MateUnmapped = 8
    ReverseCompliment = 16
    MateReverseComp = 32
    IsFirstInPair = 64
    IsMate = 128


class SamLine:
    def __init__(self, plaintext_line: str, reference_sequences):
        self.line = plaintext_line.strip().split('\t')
        self.reference_sequences = reference_sequences

        self.required_tags = {tag : self.line[tag.value] for tag in SamTags}
        self.optional_tags = {}
        for optional_tag in self.line[11:]:
            if optional_tag[:5] == 'MD:Z:':
                self.optional_tags['MD'] = optional_tag[5:]

        self.read_len = len(self.required_tags[SamTags.Read])

    def is_mapped(self):
        return (int(self.required_tags[SamTags.MapFlag]) & MapFlags.Unmapped.value) == 0

    def is_reverse_complimented(self):
        return (int(self.required_tags[SamTags.MapFlag]) & MapFlags.ReverseCompliment.value) == MapFlags.ReverseCompliment.value

    def get_fragment(self):
        map_pos = int(self[SamTags.MapPos])
        split_cigar = re.findall('\d+|\D+', self[SamTags.Cigar])
        start_clip = 0
        if split_cigar[1] == 'S':
            start_clip = int(split_cigar[0])
        reference_index = map_pos - start_clip - 1

        ref_seq_name = self[SamTags.ContigName]
        ref_seq = self.reference_sequences[ref_seq_name]
        ref_seq = self.handler.reference_sequences[ref_seq_name]

        # Match to the nearest complete window of size read_len
        if reference_index < 0:
            reference_index = 0
        if reference_index + self.read_len > len(ref_seq)-1:
            reference_index = len(ref_seq)-self.read_len-1

        frag = ref_seq[reference_index: reference_index + self.read_len]
        if not self.is_reverse_complimented():
            return frag
        else:
            return complement_seq(frag[::-1])

    def __str__(self):
        return '\t'.join(self.line)

    def __getitem__(self, key) -> str:
        if key in SamTags:
            return self.required_tags[key]
        else:
            return self.optional_tags[key]


class SamHandler:
    def __init__(self, file_path, reference_path, load_unmapped = False):
        self.file_path = file_path
        self.reference_path = reference_path
        self.reference_sequences = self.get_multifasta_sequences()
        self.unmapped_loaded = load_unmapped

        self.header = []
        self.contents: List[SamLine] = []
        with open(file_path, 'r') as f:
            for line in f:
                if line[0] == '@':
                    self.header.append(line)
                    continue
                sam_line = SamLine(line, self.reference_sequences)
                if load_unmapped or sam_line.is_mapped():
                    self.contents.append(sam_line)
        # print("Constructed handler with " + str(len(self.contents)) + " sam lines")

    def mapped_lines(self) -> Iterable[SamLine]:
        if not self.unmapped_loaded:
            yield from self.contents
        else:
            for line in self.contents:
                if line.is_mapped:
                    yield line

    def get_multifasta_sequences(self) -> Dict[str, str]:
        reference_sequences = {}
        for record in SeqIO.parse(self.reference_path, format="fasta"):
            reference_sequences[str(record.id)] = str(record.seq)
        return reference_sequences

import enum

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

'''
The mapping types given in the MapFlag tag. The actual tag is given bitwise,
so the presence of these tags is found as:
(Line[SamTags.MapFlag] & MapFlags.flag == MapFlags.flag)
'''
class MapFlags(enum.Enum):
    ReadIsPaired = 1
    Unmapped = 4
    MateUnmapped = 8
    ReverseCompliment = 16
    MateReverseComp = 32
    IsFirstInPair = 64
    IsMate = 128

class SamLine:
    def __init__(self, plaintext_line: str):
        self.line = plaintext_line.strip().split('\t')
        self.required_tags = {tag : self.line[tag.value] for tag in SamTags}
        self.optional_tags = {}
        for optional_tag in line[11:]:
            if optional_tag[:5] == 'MD:Z:':
                self.optional_tags['MD'] = optional_tag[5:]

    def is_mapped(self):
        return self.required_tags[SamTags.MapFlag] & MapFlags.Unmapped == MapFlags.Unmapped

    def get_fragment(self, reference_path):

        map_pos = int(self[SamTags.MapPos])
        split_cigar = re.findall('\d+|\D+', self[SamTags.Cigar])
        start_clip = 0
        if split_cigar[1] == 'S':
            start_clip = int(split_cigar[0])
        reference_index = map_pos - start_clip - 1

        reference_sequences = {}
        with open(reference_path, 'r') as ref_file:
            


    def __str__(self):
        return '\t'.join(self.line)

    def __getitem__(self, key):
        if key in SamTags:
            return self.required_tags[key]
        else:
            return self.optional_tags[key]


class SamHandler:
    def __init__(self, file_path, reference_path, load_unmapped = False):
        self.file_path = file_path
        self.reference_path = reference_path
        self.unmapped_loaded = load_unmapped

        self.header = []
        self.contents = []
        with open(file_path, 'r') as f:
            for line in f:
                if line[0] == '@':
                    self.header.append(line)
                    continue
                sam_line = SamLine(line)
                if load_unmapped or same_line.is_mapped():
                    self.contents.append(sam_line)

    def mapped_lines(self):
        if not self.unmapped_loaded:
            yield from self.contents
        else:
            for line in self.contents:
                if line.is_mapped:
                    yield line

from chronostrain.util.sequences import nucleotide_GAP_z4, map_nucleotide_to_z4
VCF_GAP_CHAR: str = "D"


def map_nucleotide_to_z4_with_special(x: str) -> int:
    if x == VCF_GAP_CHAR:
        return nucleotide_GAP_z4
    else:
        return map_nucleotide_to_z4(x)

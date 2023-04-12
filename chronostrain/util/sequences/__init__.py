from .base import Sequence
from .allocated import AllocatedSequence
from .dynamic_fasta import FastaIndexedResource, DynamicFastaSequence, FastaRecordNotFound
from .z4 import \
    NucleotideDtype, UnknownNucleotideError, \
    nucleotide_GAP_z4 as bytes_GAP, \
    nucleotide_N_z4 as bytes_N

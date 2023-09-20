"""
Use primers for O antigen gene cluster to pull out genes using reference fasta/GFF files.

Primers are derived from Liu, Yanhong, et al.
Liu, Yanhong, et al. "Escherichia coli O-antigen gene clusters of serogroups O62, O68, O131, O140, O142, and O163: DNA sequences and similarity between O62 and O68, and PCR-based serogrouping." Biosensors 5.1 (2015): 51-68.
"""

from typing import *
import pandas as pd
import gzip
import click
from pathlib import Path
from tqdm import tqdm

from chronostrain.util.external import call_command
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.Emboss import PrimerSearch
from BCBio import GFF


def coord_minus_to_plus(minus_coord: int, genome_len: int) -> int:
    return genome_len - minus_coord + 1


def overlaps(loc1: SimpleLocation, loc2: SimpleLocation) -> bool:
    return overlap_primitive(loc1.start, loc1.end, loc2.start, loc2.end)


def overlap_primitive(x_start, x_end, y_start, y_end) -> bool:
    return max(x_start, y_start) <= min(x_end, y_end)


def perform_primer_search(ref_seq_path: Path, in_path: Path, out_path: Path, cluster_name: str, primer1: Seq, primer2: Seq):
    # Prepare input file for `primersearch` EMBOSS program.
    primer_search_input = PrimerSearch.InputRecord()
    primer_search_input.add_primer_set(cluster_name, primer1, primer2)
    with open(in_path, 'wt') as f:
        print(str(primer_search_input), file=f)

    # Invoke the program.
    call_command(
        'primersearch',
        args=[
            '-seqall', ref_seq_path,
            '-infile', in_path,
            '-mismatchpercent', 30,
            '-outfile', out_path
        ],
    )


class PrimerSearchEmptyHits(BaseException):
    pass


def parse_primersearch_output(out_path: Path) -> Iterator[SimpleLocation]:
    with open(out_path, 'rt') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            elif line.startswith("Primer name"):
                continue
            elif line.startswith("Amplimer"):
                continue
            elif 'hits forward strand' in line:
                i = line.index('at ')
                j = line.index(' with')
                fwd_pos = int(line[i+len('at '):j].strip())
            elif 'hits reverse strand' in line:
                continue
            elif line.startswith("\tAmplimer length:"):
                # this occurs at the end of a section.
                bp_token = line.strip()[len("Amplimer length:"):-len("bp")]
                try:
                    bp_len = int(bp_token)
                    yield SimpleLocation(start=fwd_pos-1, end=fwd_pos + bp_len - 1, strand=+1)
                except ValueError:
                    raise ValueError("Couldn't parse len from token {}".format(bp_token)) from None


def get_primersearch_hit(out_path: Path, max_possible_len: int) -> SimpleLocation:
    best_loc = None
    best_len = max_possible_len
    for loc in parse_primersearch_output(out_path):
        if len(loc) < best_len:
            best_len = len(loc)
            best_loc = loc
    if best_loc is None:
        raise PrimerSearchEmptyHits()
    return best_loc


class GeneSequence:
    def __init__(self, name: str, seq: Seq):
        self.name = name
        self.seq = seq

    def __str__(self):
        return "{}:{}".format(self.name, self.seq)

    def __repr__(self):
        return "{}[seq={}]".format(self.name, self.seq)


def extract_genes(target_acc: str, seq_path: Path, gff_path: Path, target_loc: SimpleLocation) -> List[GeneSequence]:
    if gff_path.suffix == ".gz":
        f = gzip.open(gff_path, "rt")
    else:
        f = open(gff_path, "rt")

    try:
        genome_seq = SeqIO.read(seq_path, "fasta")

        chromosome_rec = None
        for seq_rec in GFF.parse(f):
            if seq_rec.id == target_acc:
                chromosome_rec = seq_rec
        if chromosome_rec is None:
            raise ValueError(f"Couldn't find target accession record `{target_acc}`")

        genes = []
        for feature in chromosome_rec.features:
            if feature.type != 'gene':
                continue
            if overlaps(feature.location, target_loc):
                feature_names = feature.qualifiers['Name']
                if len(feature_names) > 1:
                    print("[WARNING] {}, feature ID {} has multiple names: {}".format(
                        target_acc, feature.qualifiers['ID'],
                        feature_names
                    ))
                genes.append(
                    GeneSequence(
                        name=feature_names[0],
                        seq=feature.location.extract(genome_seq).seq
                    )
                )
        return genes
    finally:
        f.close()


def get_serotype_genes(
        accession: str,
        acc_chrom_path: Path,
        maximal_amplicon_len: int,
        cluster_name: str,
        primer1: Seq,
        primer2: Seq,
        gff_path: Path,
        tmp_dir: Path
) -> List[GeneSequence]:
    # =========== primersearch
    in_path = tmp_dir / 'primer_input.txt'
    out_path = tmp_dir / 'primer_output.txt'
    perform_primer_search(acc_chrom_path, in_path, out_path, cluster_name, primer1, primer2)

    # =========== parse primersearch
    target_loc = get_primersearch_hit(out_path, maximal_amplicon_len)
    # print("Best primer hit is {}, len = {}".format(target_loc, len(target_loc)))

    # =========== parse gene features.
    return extract_genes(accession, acc_chrom_path, gff_path, target_loc)


@click.command()
@click.option(
    '--index-path', '-i', 'index_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the RefSeq index catalog TSV file."
)
@click.option(
    '--out-path', '-o', 'out_path',
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="The output path, to be interpreted as a pandas Dataframe in .feather (pyarrow) format."
)
@click.option(
    '--tmp-dir', '-t', 'tmp_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="A path to a temporary directory (need not already exist). "
         "EMBOSS primersearch outputs will be temporarily stored here."
)
@click.option(
    '--primer1', '-p1', 'primer1_seq',
    type=str, required=True,
    help="The 5'--3' starting primer."
)
@click.option(
    '--primer2', '-p2', 'primer2_seq',
    type=str, required=True,
    help="The 5'--3' ending primer."
)
@click.option(
    '--cluster-name', '-n', 'cluster_name',
    type=str, required=False, default='GeneClusterTarget',
    help="A cluster name to use for EMBOSS, purely for debugging purposes."
)
@click.option(
    '--amplicon-len', '-l', 'expected_amplicon_len',
    type=int, required=False, default=None,
    help="If known, the expected length of the amplicon sequence flanked by the primers "
         "(e.g. primer for a single gene should yield the average length for that gene.)"
)
def main(
        index_path: Path,
        out_path: Path,
        tmp_dir: Path,
        cluster_name: str,
        primer1_seq: str,
        primer2_seq: str,
        expected_amplicon_len: Union[int, None]
):
    main_wrapper(index_path, out_path, tmp_dir, cluster_name,
                 primer1_seq, primer2_seq,
                 expected_amplicon_len=expected_amplicon_len)


def main_wrapper(
        index_path: Path,
        out_path: Path,
        tmp_dir: Path,
        cluster_name: str,
        primer1_seq: str,
        primer2_seq: str,
        expected_amplicon_len: Union[int, None]
):
    tmp_dir.mkdir(exist_ok=True, parents=True)
    index_df = pd.read_csv(index_path, sep='\t')

    primer1 = Seq(primer1_seq)
    primer2 = Seq(primer2_seq)

    df_entries = []
    for _, row in tqdm(index_df.iterrows(), total=index_df.shape[0]):
        acc = row['Accession']
        ref_seq_path = Path(row['SeqPath'])
        chrom_len = row['ChromosomeLen']
        gff_path = Path(row['GFF'])
        genus = row['Genus']
        species = row['Species']
        strain = row['Strain']
        if expected_amplicon_len is not None:
            max_amplicon_len = 1.5 * expected_amplicon_len
        else:
            max_amplicon_len = chrom_len

        try:
            genes = get_serotype_genes(
                acc,
                ref_seq_path,
                max_amplicon_len,
                cluster_name,
                primer1,
                primer2,
                gff_path,
                tmp_dir
            )
        except PrimerSearchEmptyHits:
            print(f"Couldn't find primer-based hits for {acc} ({genus} {species}, Strain {strain})")
            continue
        if len(genes) > 50:
            print("[WARNING] {} ({} {}, Strain {}) has {} genes (more than the threshold of 30)".format(
                acc, genus, species, strain,
                len(genes)
            ))
            continue

        for gene in genes:
            df_entries.append({
                'Accession': acc,
                'Gene': gene.name,
                'GeneSeq': str(gene.seq)
            })
    df = pd.DataFrame(df_entries)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_feather(out_path)


if __name__ == "__main__":
    main_wrapper(
        index_path=Path("/mnt/e/ecoli_db/ref_genomes/index.tsv"),
        out_path=Path("/mnt/e/ecoli_db/serotype_O_genes/result.feather"),
        tmp_dir=Path('/home/youn/work/emboss_test'),
        cluster_name='O-antigen_JUMPSTART_GND_1',
        primer1_seq='CATGGTAGCTGTAAAGCCAGGGGCGGTAGCGTG',  # 5'--3' convention; JUMPstart sequence
        primer2_seq='CATGCTGCCATACCGACGACGCCGATCTGTTGCTTKGACA',  # 5'--3' convention; GND primer
        expected_amplicon_len=None
    )

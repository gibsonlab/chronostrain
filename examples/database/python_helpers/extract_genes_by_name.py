from typing import *
from pathlib import Path

import click
import pandas as pd

import gzip
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from BCBio import GFF


def gene_name_is_hit(feature_names: List[str], target_prefixes: List[str]) -> Tuple[bool, str]:
    for gene_name_candidate in feature_names:
        for prefix in target_prefixes:
            if gene_name_candidate.startswith(prefix):
                return True, gene_name_candidate
    return False, ""


class GeneSequence:
    def __init__(self, name: str, seq: Seq):
        self.name = name
        self.seq = seq

    def __str__(self):
        return "{}:{}".format(self.name, self.seq)

    def __repr__(self):
        return "{}[seq={}]".format(self.name, self.seq)


def extract_genes(target_acc: str, gff_path: Path, seq_path: Path, gene_name_prefixes: List[str]) -> List[GeneSequence]:
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

            feature_names = feature.qualifiers['Name']
            is_hit, gene_name = gene_name_is_hit(feature_names, gene_name_prefixes)
            if is_hit:
                genes.append(
                    GeneSequence(
                        name=gene_name,
                        seq=feature.location.extract(genome_seq).seq
                    )
                )
        return genes
    finally:
        f.close()


@click.command()
@click.option(
    '--index-path', '-i', 'index_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the RefSeq index catalog TSV file."
)
@click.option(
    '--genus', '-g', 'target_genus',
    type=str, required=True, help='The name of the target genus.'
)
@click.option(
    '--species', '-s', 'target_species',
    type=str, required=True, help='The name of the target species.'
)
@click.option(
    '--out-path', '-o', 'out_path',
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="The output path, to be interpreted as a pandas Dataframe in .feather (pyarrow) format."
)
@click.option(
    '--gene-name-prefixes', '-p', 'gene_name_prefixes',
    multiple=True,
    help="The gene name prefix to search for. "
         "For example, `fim` specifies a search for any fim gene: fimA, fimB, fimC, etc. "
         "Argument can be repeated to specify multiple patterns (-g pattern1 -g pattern2 ...)"
)
def main(
        index_path: Path,
        target_genus: str,
        target_species: str,
        gene_name_prefixes: List[str],
        out_path: Path
):
    index_df = pd.read_csv(index_path, sep='\t')
    index_df = index_df.loc[
        (index_df['Genus'].str.lower() == target_genus.lower())
        & (index_df['Species'].str.lower() == target_species.lower())
    ]

    df_entries = []
    for _, row in tqdm(index_df.iterrows(), total=index_df.shape[0], unit='genome'):
        acc = row['Accession']
        gff_path = Path(row['GFF'])
        seq_path = Path(row['SeqPath'])
        genes = extract_genes(acc, gff_path, seq_path, gene_name_prefixes)
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
    main()

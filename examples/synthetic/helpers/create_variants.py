from dataclasses import dataclass
from typing import Dict, List
import argparse
import json
from pathlib import Path

from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_args():
    parser = argparse.ArgumentParser(description="Create variant genomes using specified input file. "
                                                 "Generates a brand-new JSON database file for sampling purposes.")

    parser.add_argument('-i', '--input_variants', dest='variant_json_path', required=True, type=str,
                        help='<Required> The input JSON file that specifies a list of variants to create.')
    parser.add_argument('-o', '--output_dir', required=True, type=str,
                        help='<Required> The target output directory.')

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(args.variant_json_path, "r") as f:
        parse_variants(json.load(f), output_dir)


# =================================== Aux functions =====================================
@dataclass
class Insertion:
    pos: int
    seq: str


@dataclass
class Deletion:
    pos: int
    len: int


@dataclass
class Substitution:
    pos: int
    base: str


def parse_variants(variant_desc_list: List[Dict], output_dir: Path):
    for idx, desc in enumerate(variant_desc_list):
        yield parse_variant_desc(desc, output_dir)


def download_fasta(accession: str, out_path: Path):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    print("[{}] Downloading entrez file ({})...".format(
        accession[0] if isinstance(accession, list) else accession,
        str(out_path.name)
    ))
    net_handle = Entrez.efetch(db='nucleotide', id=accession, rettype='fasta', retmode='text')
    with open(out_path, "w") as f:
        f.write(net_handle.read())
    net_handle.close()

    print("[{ac}] download completed.".format(
        ac=accession[0] if isinstance(accession, list) else accession
    ))


def parse_variant_desc(variant_desc: Dict, output_dir: Path):
    src_strain_id = variant_desc['source']
    variant_id = variant_desc['target']

    fasta_path = output_dir / src_strain_id / f'{src_strain_id}.fasta'
    download_fasta(src_strain_id, fasta_path)

    genome = str(SeqIO.read(fasta_path, "fasta").seq)

    insertions = [
        Insertion(int(d['pos']), d['seq'])
        for d in variant_desc.get('insertions', [])
    ]
    deletions = [
        Deletion(int(d['pos']), int(d['len']))
        for d in variant_desc.get('deletions', [])
    ]
    substitutions = [
        Substitution(int(d['pos']), d['base'])
        for d in variant_desc.get('substitutions', [])
    ]

    # ========= Save altered genome.
    variant_genome = apply_variations(genome, insertions, deletions, substitutions)
    fasta_path = output_dir / variant_id / f"{variant_id}.fasta"
    record = SeqRecord(Seq(variant_genome), id=variant_id, description="")
    SeqIO.write([record], fasta_path, format='fasta')
    print("Created variant FASTA file {}".format(fasta_path))


def apply_variations(genome: str,
                     insertions: List[Insertion],
                     deletions: List[Deletion],
                     substitutions: List[Substitution]) -> str:
    seq = [x for x in genome]
    for insertion in insertions:
        print(f"Applying INSERTION of {insertion.seq.upper()} at {insertion.pos}")
        seq[insertion.pos - 1] = seq[insertion.pos - 1] + insertion.seq.upper()

    for deletion in deletions:
        print(f"Applying DELETION of {deletion.len} chars at {deletion.pos}")
        for pos in range(deletion.pos, deletion.pos + deletion.len):
            seq[pos - 1] = ""

    for substitution in substitutions:
        print(f"Applying SUBSTITUTION of {substitution.base.upper()} "
              f"with {seq[substitution.pos - 1]} at {substitution.pos}")
        seq[substitution.pos - 1] = substitution.base.upper()

    return "".join(seq)


if __name__ == "__main__":
    main()



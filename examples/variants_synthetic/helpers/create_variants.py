from dataclasses import dataclass
from typing import Iterator, Dict, List, Tuple
import argparse
import json
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.database import StrainDatabase
from chronostrain.config import cfg


# ==================================== Main entry ======================================
from chronostrain.model import Strain
from chronostrain.config import create_logger
logger = create_logger("chronostrain.create_variant")


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
    db = cfg.database_cfg.get_database(load_full_genomes=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(args.variant_json_path, "r") as f:
        for variant_id, variant_seq in parse_variants(db, json.load(f)):
            fasta_path = output_dir / f"{variant_id}.fasta"
            record = SeqRecord(
                Seq(variant_seq),
                id=variant_id,
                description=""
            )
            SeqIO.write(
                [record], fasta_path, format='fasta'
            )
            logger.info("Created variant FASTA file {}".format(
                fasta_path
            ))


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


def parse_variants(db: StrainDatabase, variant_desc_list: List[Dict]) -> Iterator[Strain]:
    for idx, desc in enumerate(variant_desc_list):
        yield parse_variant_desc(db, desc)


def parse_variant_desc(db: StrainDatabase, variant_desc: Dict) -> Tuple[str, str]:
    strain_id = variant_desc['source']
    variant_id = variant_desc['target']
    strain = db.get_strain(strain_id)
    genbank_path = strain.metadata.genbank_path

    record = next(SeqIO.parse(genbank_path, "genbank"))
    genome = str(record.seq)

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

    return variant_id, apply_variations(genome, insertions, deletions, substitutions)


def apply_variations(genome: str,
                     insertions: List[Insertion],
                     deletions: List[Deletion],
                     substitutions: List[Substitution]) -> str:
    seq = [x for x in genome]
    for insertion in insertions:
        logger.info(f"Applying INSERTION of {insertion.seq.upper()} at {insertion.pos}")
        seq[insertion.pos - 1] = seq[insertion.pos - 1] + insertion.seq.upper()

    for deletion in deletions:
        logger.info(f"Applying DELETION of {deletion.len} chars at {deletion.pos}")
        for pos in range(deletion.pos, deletion.pos + deletion.len):
            seq[pos - 1] = ""

    for substitution in substitutions:
        logger.info(f"Applying SUBSTITUTION of {substitution.base.upper()} with {seq[substitution.pos - 1]} at {substitution.pos}")
        seq[substitution.pos - 1] = substitution.base.upper()

    return "".join(seq)


if __name__ == "__main__":
    main()



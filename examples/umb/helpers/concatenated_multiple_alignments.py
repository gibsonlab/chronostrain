import argparse
from pathlib import Path
from typing import List, Dict, Set

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.database import JSONStrainDatabase, StrainDatabase
from chronostrain.model import Marker
from chronostrain.util.external import mafft_global

from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger("chronostrain.multiple_alignments")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster strains by similarity (in hamming distance of concatenated marker alignments)"
    )

    # Input specification.
    parser.add_argument('--raw_json', required=True, type=str,
                        help='<Required> The raw database JSON file, PRIOR to resolving overlaps.')
    parser.add_argument('--align_path', required=True, type=str,
                        help='<Required> The target alignment file to output to.')

    parser.add_argument('-u', '--uniprot_csv', required=False, type=str, default='',
                        help='<Optional> A path to a two-column CSV file (<UniprotID>, <ClusterName>) format specifying'
                             'any desired additional genes not given by metaphlan.')
    parser.add_argument('-m3', '--metaphlan3_pkl', required=False, type=str, default='',
                        help='<Optional> A path to the metaphlan3 database pickle file.')
    parser.add_argument('-m4', '--metaphlan4_pkl', required=False, type=str, default='',
                        help='<Optional> A path to the metaphlan4 database pickle file.')
    parser.add_argument('-c', '--clermont_fasta', required=False, type=str, default='',
                        help='<Optional> A path to a fasta file listing out genes. Each records ID must be the '
                             'desired gene name.')
    parser.add_argument('--marker_choice', required=False, type=str, default="all",
                        help='<Optional> A string specifying which group of markers to choose. '
                             'Supported options are: [all, metaphlan, mlst, clermont] '
                             '(Default: `all`).')
    return parser.parse_args()


def multi_align_markers(output_path: Path, markers: List[Marker], n_threads: int = 1) -> Dict[str, SeqRecord]:
    input_fasta_path = output_path.with_suffix('.input.fasta')

    SeqIO.write(
        [marker.to_seqrecord() for marker in markers],
        input_fasta_path,
        "fasta"
    )

    mafft_global(
        input_fasta_path=input_fasta_path,
        output_path=output_path,
        n_threads=n_threads,
        auto=True,
        max_iterates=1000
    )

    ids_to_records = {}
    for record in SeqIO.parse(output_path, format='fasta'):
        strain_id, marker_name, marker_id = Marker.parse_seqrecord_id(record.id)
        ids_to_records[marker_id] = record
    return ids_to_records


def get_all_alignments(db: StrainDatabase, work_dir: Path, marker_names: Set[str]) -> Dict[str, Dict[str, SeqRecord]]:
    work_dir.mkdir(exist_ok=True, parents=True)
    all_alignments = {}
    for gene_name in marker_names:
        alignment_records = multi_align_markers(
            output_path=work_dir / f"{gene_name}.fasta",
            markers=db.get_markers_by_name(gene_name),
            n_threads=cfg.model_cfg.num_cores
        )

        all_alignments[gene_name] = alignment_records
    return all_alignments


def get_concatenated_alignments(db: StrainDatabase, out_path: Path, marker_names: List[str]):
    """
    Generates a single FASTA file containing the concatenation of the multiple alignments of each marker gene.
    If a gene is missing from a strain, gaps are appended instead.
    If multiple hits are found, then the first available one is used (found in the same order as BLAST hits).
    """
    all_marker_alignments = get_all_alignments(db, out_path.parent / "marker_multiple_alignments", set(marker_names))

    records: List[SeqRecord] = []
    for strain in db.all_strains():
        seqs_to_concat = []

        # Remember the specific marker to extract alignment from, for this particular strain.
        strain_marker_map: Dict[str, Marker] = {}
        for marker in strain.markers:
            if marker.name not in marker_names:
                continue

            # Using marker that comes first in the listing (usually the highest idty blast hit)"
            if marker.name in strain_marker_map:
                continue

            strain_marker_map[marker.name] = marker

        # Concatenate the alignment sequence in the particular order.
        target_gene_ids = []
        for gene_name in marker_names:
            record_map = all_marker_alignments[gene_name]

            if gene_name not in strain_marker_map:
                _, example_record = next(iter(record_map.items()))
                aln_len = len(example_record.seq)
                seqs_to_concat.append(
                    "".join('-' for _ in range(aln_len))
                )
                target_gene_ids.append("-")
            else:
                target_marker = strain_marker_map[gene_name]
                record = record_map[target_marker.id]
                seqs_to_concat.append(
                    str(record.seq)
                )
                target_gene_ids.append(target_marker.id)
        records.append(
            SeqRecord(
                Seq("".join(seqs_to_concat)),
                id=strain.id,
                description=f"{strain.name}:" + "|".join(target_gene_ids)
            )
        )

    SeqIO.write(
        records, out_path, "fasta"
    )


def extract_metaphlan_markers(pkl_path: Path) -> Set[str]:
    import bz2
    import pickle

    logger.info(f"Searching for E.coli markers from MetaPhlAn database: {pkl_path.stem}.")
    with bz2.open(pkl_path, "r") as f:
        db = pickle.load(f)

    return set(
        marker_key
        for marker_key, marker_dict in db['markers'].items()
        if 's__Escherichia_coli' in marker_dict['taxon']
    )


def get_marker_choice(db: StrainDatabase,
                      marker_choice: str,
                      uniprot_csv_path: Path,
                      metaphlan3_pkl_path: Path,
                      metaphlan4_pkl_path: Path,
                      clermont_genes_path: Path) -> List[str]:
    if marker_choice == "all":
        return sorted(db.all_marker_names())
    elif marker_choice == "metaphlan3":
        if not metaphlan3_pkl_path.exists():
            raise FileNotFoundError(
                f"If specifying marker_choice == `metaphlan3`, then a valid metaphlan pkl path must be provided. "
                f"(Got: {metaphlan3_pkl_path})"
            )
        return sorted(extract_metaphlan_markers(metaphlan3_pkl_path))
    elif marker_choice == "metaphlan4":
        if not metaphlan4_pkl_path.exists():
            raise FileNotFoundError(
                f"If specifying marker_choice == `metaphlan4`, then a valid metaphlan pkl path must be provided. "
                f"(Got: {metaphlan4_pkl_path})"
            )
        return sorted(extract_metaphlan_markers(metaphlan4_pkl_path))
    elif marker_choice == "mlst":
        if not uniprot_csv_path.exists():
            raise FileNotFoundError(
                f"If specifying marker_choice == `mlst`, then a valid CSV file that includes MLST genes must be provided. "
                f"(Got: {uniprot_csv_path})"
            )

        target_genes = set()
        with open(uniprot_csv_path, "r") as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue

                tokens = line.strip().split("\t")
                uniprot_id, gene_name, metadata = tokens[0], tokens[1], tokens[2]

                if uniprot_id == "UNIPROT_ID":
                    # Header line
                    continue

                if "MLST" in metadata:
                    target_genes.add(gene_name)
        return sorted(target_genes)
    elif marker_choice == "clermont":
        if not clermont_genes_path.exists():
            raise FileNotFoundError(
                f"If specifying marker_choice == `clermont`, then a valid FASTA path must be provided. "
                f"(Got: {clermont_genes_path})"
            )

        return sorted({
            record.id
            for record in SeqIO.parse(clermont_genes_path, 'fasta')
        })
    else:
        raise ValueError(f"Unrecognized marker_choice string `{marker_choice}`")


def main():
    args = parse_args()

    raw_json_path = Path(args.raw_json)
    raw_db = JSONStrainDatabase(
        entries_file=raw_json_path,
        data_dir=cfg.database_cfg.data_dir,
        marker_max_len=cfg.database_cfg.db_kwargs['marker_max_len'],
        force_refresh=False
    )

    marker_names = get_marker_choice(raw_db,
                                     args.marker_choice,
                                     Path(args.uniprot_csv),
                                     Path(args.metaphlan3_pkl),
                                     Path(args.metaphlan4_pkl),
                                     Path(args.clermont_fasta))
    get_concatenated_alignments(raw_db, Path(args.align_path), marker_names)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        raise

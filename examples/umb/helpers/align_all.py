import argparse
import glob
from pathlib import Path
from typing import List, Tuple

from chronostrain.database import StrainDatabase
from chronostrain.model import SequenceRead
from chronostrain.util.alignments import multiple, pairwise
from chronostrain.util.alignments.sam import SamFile
from chronostrain.config import cfg, create_logger
logger = create_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create multiple alignment of marker genes to read fragments."
    )

    # Input specification.
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='<Required> The path (or glob pattern, e.g. `*.fastq`) to the input fastq files '
                             'containing reads.')
    parser.add_argument('-o', '--out_path', required=True, type=str,
                        help='<Required> The target output path for the marker gene multiple alignment (fasta format).')
    parser.add_argument('-m', '--marker_name', required=True, type=str,
                        help='<Required> The marker gene name to specify.')
    parser.add_argument('-w', '--work_dir', required=False, type=str,
                        default='',
                        help='<Required> A directory to store temporary files. By default, create a `tmp`'
                             'directory in the target output path.')

    parser.add_argument('-t', '--threads', required=False, type=int,
                        default=cfg.model_cfg.num_cores,
                        help='<Optional> The number of threads to use.')
    return parser.parse_args()


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    if len(args.work_dir) == 0:
        work_dir = out_path.parent / "tmp"
    else:
        work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)

    temp_fasta_path = work_dir / f"tmp_{args.marker_name}_align.fasta"

    multiple.align(
        db,
        marker_name=args.marker_name,
        read_descriptions=read_descriptions(
            db,
            pathname=args.input,
            work_dir=work_dir,
            marker_name=args.marker_name
        ),
        intermediate_fasta_path=temp_fasta_path,
        out_fasta_path=out_path,
        n_threads=args.threads
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.error(e)
        raise


# ============================================== Function definitions
def read_descriptions(
        db: StrainDatabase,
        pathname: str,
        work_dir: Path,
        marker_name: str
) -> List[Tuple[SequenceRead, bool]]:
    desc = []
    for read_path in glob.glob(pathname):
        read_path = Path(read_path)
        sam_path = work_dir / f"{read_path.with_suffix('').name}.sam"

        # Perform pairwise alignment.
        pairwise.BowtieAligner(
            reference_path=db.multifasta_file,
            index_basepath=db.multifasta_file.parent,
            index_basename="markers",
            num_threads=cfg.model_cfg.num_cores
        ).align(query_path=read_path, output_path=sam_path)

        for pair_aln in pairwise.parse_alignments(SamFile(sam_path, 'fastq'), db):
            if pair_aln.marker.name != marker_name:
                continue

            desc.append((pair_aln.read, pair_aln.reverse_complemented))
    return desc

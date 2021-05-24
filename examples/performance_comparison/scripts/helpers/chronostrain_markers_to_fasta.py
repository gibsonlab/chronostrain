from pathlib import Path
import argparse
from chronostrain import cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="For each strain in the database, save all of its markers into a single multi-fasta file."
    )

    # Input specification.
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        required=True, type=str,
                        help='<Required> The output directory.')
    parser.add_argument('-e', '--extension', dest="extension", required=False, type=str,
                        default=".markers.fasta",
                        help='<Optional> The file extension to save the markers with. (Default: .markers.fasta)')
    return parser.parse_args()


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()
    for strain in db.all_strains():
        marker_out_path = Path(args.output_dir) / "{}{}".format(strain.id, args.extension)
        db.strain_markers_to_fasta(
            strain_id=strain.id,
            out_path=marker_out_path,
            file_mode="w"
        )


if __name__ == "__main__":
    main()

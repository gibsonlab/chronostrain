import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create bash script wrapper for clermonTyping.sh."
    )

    # Input specification.
    parser.add_argument('-i', '--strainge_db_dir', required=True, type=str,
                        help='<Required> The path to the strainge database directory containing strain-named fasta '
                             'files (usually together with .hdf5 k-mer files).')
    parser.add_argument('-c', '--clermon_script_path', required=True, type=str,
                        help='<Required> The path to clermonTyping.sh')
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The target output path to write the resulting wrapper script to.')

    parser.add_argument('-a', '--analysis_name', required=False, type=str,
                        default='umb',
                        help='<Optional> The name of the analysis to pass to clermonTyping. (default: `umb`)')
    return parser.parse_args()


def main():
    args = parse_args()
    straingst_db_path = Path(args.strainge_db_dir)
    fasta_paths = straingst_db_path.glob("*.fa.gz")

    script = "bash {clermon_script_path} --fasta {fasta_path} --name {analysis_name}"

    script = script.format(
        clermon_script_path=args.clermon_script_path,
        fasta_path='@'.join(str(p) for p in fasta_paths),
        analysis_name=args.analysis_name
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        print(script, file=f)

    print(f"Wrote script wrapper to {args.output_path}")


if __name__ == "__main__":
    main()

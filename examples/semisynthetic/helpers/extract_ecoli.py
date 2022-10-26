from pathlib import Path
import json
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_json', type=str, required=True)
    parser.add_argument('-o', '--output_json', type=str, required=True)
    return parser.parse_args()


def restrict_ecoli(in_path: Path, out_path: Path):
    with open(in_path, 'r') as f:
        entries = json.load(f)

    filtered_entries = []
    for entry in entries:
        if entry['genus'] == 'Escherichia' and entry['species'] == 'coli':
            filtered_entries.append(entry)

    with open(out_path, 'w') as f:
        json.dump(filtered_entries, f)


def main():
    args = parse_args()
    in_path = Path(args.input_json)
    out_path = Path(args.output_json)
    restrict_ecoli(in_path, out_path)


if __name__ == "__main__":
    main()

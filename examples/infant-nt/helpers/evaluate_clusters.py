import argparse
from pathlib import Path
import json
from typing import Dict
import pandas as pd


def parse_mlst_output(mlst_path: Path) -> Dict[str, str]:
    st_dict = {}
    with open(mlst_path, "rt") as f:
        for line in f:
            tokens = line.strip().split('\t')
            fname = tokens[0]
            acc = fname[:-len(".fasta")]
            scheme_name = tokens[1]
            st_id = tokens[2]
            st_dict[acc] = f'{scheme_name}:{st_id}'
    return st_dict


def evaluate(json_path: Path, mlst_path: Path) -> pd.DataFrame:
    with open(json_path, "rt") as json_f:
        db = json.load(json_f)

    st_dict = parse_mlst_output(mlst_path)

    df_entries = []
    for strain in db:
        representative_acc = strain['id']
        cluster = strain['cluster']
        for member in cluster:
            member_acc = member.split("(")[0]
            df_entries.append({
                'Cluster': representative_acc,
                'Member': member_acc,
                'ST': st_dict[member_acc] if member_acc in st_dict else "UNKNOWN"
            })
    return pd.DataFrame(df_entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--mlst', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json)
    mlst_output_path = Path(args.mlst)
    st_df = evaluate(json_path, mlst_output_path)
    st_df.to_csv(mlst_output_path.parent / "database_sts.csv", index=False)


if __name__ == "__main__":
    main()

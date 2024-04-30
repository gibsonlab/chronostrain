import sys
from pathlib import Path
import pandas as pd


def main():
    table_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not table_path.exists():
        exit(1)

    df = pd.read_csv(table_path, sep='\t', header=0)
    df = df.loc[~df['sample_title'].str.contains('_M')]
    df['Participant'] = df['sample_title'].apply(extract_patient_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wt") as out_f:
        for participant_id in sorted(df['Participant'].unique()):
            print(participant_id, file=out_f)


def extract_patient_name(sample_name: str):
    tokens = sample_name.split('_')
    if len(tokens) > 1 and tokens[1].startswith('T'):  # is a twin
        return "{}_{}".format(tokens[0], tokens[1])
    else:
        return tokens[0]


if __name__ == "__main__":
    main()

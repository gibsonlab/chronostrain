import sys
from pathlib import Path
import pandas as pd


def main():
    table_path = Path(sys.argv[1])
    if not table_path.exists():
        exit(1)

    df = pd.read_csv(table_path, sep='\t', header=0)
    df['Participant'] = df['sample_title'].apply(extract_patient_name)
    for participant_id in sorted(df['Participant'].unique()):
        print(participant_id)


    # for participant_id in df['individual'].unique():
    #     print(participant_id)



def extract_patient_name(sample_name: str):
    tokens = sample_name.split('_')

    if len(tokens) > 1 and tokens[1].startswith('T'):  # is a twin
        return "{}_{}".format(tokens[0], tokens[1])
    else:
        return tokens[0]


if __name__ == "__main__":
    main()

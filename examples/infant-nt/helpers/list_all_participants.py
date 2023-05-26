import sys
from pathlib import Path
import pandas as pd


def main():
    table_path = Path(sys.argv[1])
    if not table_path.exists():
        exit(1)

    df = pd.read_csv(table_path, sep='\t', header=0)
    for participant_id in df['individual'].unique():
        print(participant_id)


    # dataset_catalog_path = Path(sys.argv[1])
    # if not dataset_catalog_path.exists():
    #     exit(1)
    #
    # dataset = pd.read_csv(dataset_catalog_path, sep='\t')
    # dataset = dataset.sort_values('sample_title')
    # participants_seen = set()
    #
    # for _, row in dataset.iterrows():
    #     sample_title = row['sample_title']
    #
    #     tokens = sample_title.split('_')
    #     if len(tokens) == 2:
    #         participant, timepoint = tokens
    #     elif len(tokens) == 3:
    #         participant, timepoint, sample_id = tokens
    #     else:
    #         continue
    #
    #     if participant not in participants_seen:
    #         print(participant)
    #         participants_seen.add(participant)


if __name__ == "__main__":
    main()

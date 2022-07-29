from typing import Tuple
from pathlib import Path
import argparse
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--chronostrain_dir', required=True, type=str)
    parser.add_argument('-o', '--output', required=True, type=str)
    return parser.parse_args()


def umb_outputs(base_dir: Path) -> Tuple[str, torch.Tensor]:
    for umb_dir in base_dir.glob("UMB*"):
        if not umb_dir.is_dir():
            raise RuntimeError(f"Expected child `{umb_dir}` to be a directory.")
        umb_id = umb_dir.name

        sample_path = umb_dir / "samples.pt"
        if not sample_path.exists():
            print(f"File `{sample_path}` not found. Skipping {umb_id}...")

        samples = torch.load(umb_dir / "samples.pt")
        yield umb_id, samples


def evaluate(chronostrain_output_dir: Path) -> pd.DataFrame:
    df_entries = []
    for patient, umb_samples in umb_outputs(chronostrain_output_dir):
        df_entries.append({
            "Patient": patient,
            "Dominance": dominance_switch_ratio(umb_samples)
        })
    return pd.DataFrame(df_entries)


def dominance_switch_ratio(abundance_est: torch.Tensor) -> float:
    """
    Calculate how often the dominant strain switches.
    """
    medians = torch.median(abundance_est, dim=1).values
    dom = torch.argmax(medians, dim=1).numpy()
    num_switches = 0

    def row_is_zeros(r) -> bool:
        return torch.sum(r == 0).item() == r.shape[0]

    for i in range(len(dom) - 1):
        switched = (dom[i] != dom[i+1]) or row_is_zeros(medians[i]) or row_is_zeros(medians[i+1])
        if switched:
            num_switches += 1
    return num_switches / (len(dom) - 1)


def main():
    args = parse_args()
    df = evaluate(Path(args.chronostrain_dir))
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

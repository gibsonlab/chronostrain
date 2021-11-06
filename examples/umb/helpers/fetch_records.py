"""
Merges the UMB record file containing sample id/dates/sample type,
together with the SRA record IDs. (One-time use to generate umb_samples.csv.)
"""
import pandas as pd


def fetch_sra_csv(bioproject_id: str) -> pd.DataFrame:
    url = f"http://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?save=efetch&rettype=runinfo&db=sra&term=${bioproject_id}"
    return pd.read_csv(url)


def fetch_umb_tsv(tsv_path: str) -> pd.DataFrame:
    return pd.read_csv(tsv_path, sep='\t', parse_dates=['date'])


def fetch_records(bioproject_id: str, tsv_path: str) -> pd.DataFrame:
    sra = fetch_sra_csv(bioproject_id)
    umb = fetch_umb_tsv(tsv_path)
    return pd.merge(
        sra, umb,
        how='inner',
        left_on='SampleName', right_on='sample'
    )


def main():
    bioproject = "PRJNA400628"
    tsv_path = "/mnt/d/Projects/chronostrain/examples/umb/files/umb_samples_broad.tsv"
    out_path = "/mnt/d/Projects/chronostrain/examples/umb/files/umb_samples.csv"

    from datetime import date
    all_records = fetch_records(bioproject, tsv_path)
    all_records['epoch'] = date(2015, 1, 1)
    all_records['epoch'] = pd.to_datetime(all_records['epoch'])
    all_records['days'] = (all_records['date'] - all_records['epoch']).dt.days
    all_records[['Run', 'ID', 'SampleName', 'date', 'days', 'type']].to_csv(
        out_path,
        sep=',',
        index=False
    )


if __name__ == "__main__":
    main()

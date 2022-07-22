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
    tsv_path = "F:/microbiome_tracking/examples/umb/files/umb_samples_broad.tsv"
    out_path = "F:/microbiome_tracking/examples/umb/files/umb_samples.csv"
    test_group = ['UMB05', 'UMB08', 'UMB11', 'UMB12', 'UMB15', 'UMB18', 'UMB20', 'UMB23', 'UMB24']

    from datetime import date
    all_records = fetch_records(bioproject, tsv_path)
    all_records['epoch'] = date(2015, 1, 1)
    all_records['epoch'] = pd.to_datetime(all_records['epoch'])
    all_records['days'] = (all_records['date'] - all_records['epoch']).dt.days
    all_records['Group'] = 'Control'
    all_records.loc[all_records['ID'].isin(test_group), 'Group'] = 'Test'
    all_records.loc[
        (all_records['LibraryStrategy'] == 'WGS'),
        ['Run', 'ID', 'SampleName', 'date', 'days', 'type', 'Model', 'LibraryStrategy', 'Group']
    ].to_csv(
        out_path,
        sep=',',
        index=False
    )


if __name__ == "__main__":
    main()

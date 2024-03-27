import sys
import pandas as pd
from pathlib import Path


# Assume poppunk already ran.
ref_dir = Path(sys.argv[1])
ref_clust_file = ref_dir / "poppunk" / "threshold" / "threshold_clusters.csv"
poppunk_input_tsv = ref_dir / "poppunk" / "input.tsv"


clust_df = pd.read_csv(ref_clust_file, sep=',')  # has headers ['Taxon', 'Cluster']
input_df = pd.read_csv(poppunk_input_tsv, sep='\t', header=None).rename(columns={0: 'Taxon', 1: 'FastaPath'})



assert clust_df.shape[0] == input_df.shape[0]  # ensure no taxa got left out somehow.
merged_df = clust_df.merge(input_df, on='Taxon', how='inner')


# output files
ref_tsv_file = ref_dir / "ref_info.tsv"


merged_df[['Taxon', 'Cluster', 'FastaPath']].rename(columns={
    'Taxon': 'id',
    'Cluster': 'cluster',
    'FastaPath': 'assembly'
}).to_csv(ref_tsv_file, index=False, sep='\t', header=True)

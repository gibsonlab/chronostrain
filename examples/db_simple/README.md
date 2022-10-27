# Summary

This contains an example of a database generation pipeline; there are multiple ways to 
arrive at the end product. 
For instance, if you have high-quality MAGs (e.g. maybe you deeply sequenced a subset of your samples) 
then those should be used instead of downloading a reference genome.

In particular, we provide scripts that sets up a RefSeq index, which is a prerequisite for the 
command `chronostrain make-db`.
The steps outlined here closely resembles what was done for the analyses done in the paper.

# Requirements
- The user must separately install the tool `ncbi-genome-download`, which can be obtained from the original
author's <a href="https://github.com/kblin/ncbi-genome-download">github repo</a> or through
<a href="https://anaconda.org/bioconda/ncbi-genome-download">conda</a>.
- Since we will be extracting marker seeds from MetaPhlan (we will use version 4 as an example), the user should
also install <a href="https://huttenhower.sph.harvard.edu/metaphlan/">MetaPhlAn</a> and know where the 
database files are located. 
  
  By default (as of Oct. 2022), one runs `metaphlan --install`, which downloads/extracts a tarball into 
  `<PYTHON_LIBS>/site-packages/metaphlan/metaphlan_databases`.
  

# How to use

## Step 1: Download and compile a RefSeq index
Run the script `download_ncbi.sh`:
```
bash download_ncbi.sh
```
You may see some errors (e.g. `No entry for file ending in <..>`) but these can mostly be ignored.

The recipe outlined in that shell script is as follows:
1. Use `ncbi-genome-download` to download all complete chromosomal assemblies of Klebsiella from NCBI.
2. Run the custom python script `index_refseq.py` to create a TSV-formatted index of fasta/gff files 
   (see the main README for details), meant to serve as an input to `chronostrain make-db`.
   
## Step 2: Generate marker sequence seeds

### Strategy
In this example, we will extract all marker genes from MetaPhlAn (version 3 or later) that are assigned the
`s__Klebsiella_pneumoniae` taxonomic label, and use the provided reference FASTA records for ChronoStrain's marker seeds.
In this scenario, this database will be specialized for differentiating K. pneumoniae-specific SNVs 
(but the RefSeq index includes all assemblies of Klebsiella on NCBI, so ChronoStrain will also look in non-pneumoniae species).

```bash
python extract_metaphlan_markers.py -t s__Klebsiella_pneumoniae \
  -i <python_site_packages>/metaphlan/metaphlan_databases/mpa_vJan21_CHOCOPhlAnSGB_202103.pkl \
  -o marker_seeds
```

### Key Assumptions
Note that MetaPhlAn marker genes are specifically designed to be *core* and *clade-specific*.
Therefore, (at least in this example) we will not be extending a search (mapping marker seeds to whole genomes) outside
the Klebsiella genus.
However, if one includes *non-core* marker seeds (non-universal and non-exclusive to the clade), then they must plan 
ahead to avoid confounded strain calls.

In short, the user must ensure that the original database is expansive enough so that the following is a reasonable assumption:
**If a read passes `chronostrain filter`, then the database includes all reasonable potential labels (or 
representatives of labels, since we will be doing some clustering of genomes).**

What is "reasonable" depends on the database parametrization.
Our recommendation is to expand Step 1 to download assemblies at the family level or higher, if one has reason to suspect
homology or even >K% sequence identity.
Here, K is the % identity threshold to use for determining marker sequences:
`chronostrain make-db --min-pct-idty K`.
By default, K is 75 (allowing for considerable single-nucleotide or structural variants within a species), so if 75% match identity (with near 100% query coverage) hits appear through BLAST when querying 
the marker seed, then that clade ought to also be included.

## Step 3: Use marker sequence seeds to create ChronoStrain DB

First, one needs to compile the reference sequences in the index into a BLAST database:
```bash
bash create_blast_db.sh
```
which simply concatenates all FASTA records in the RefSeq index and invokes `makeblastdb` to create
a database called `kleb_ex`.

Then, we invoke chronostrain:
```bash
mkdir chronostrain-db
chronostrain -c chronostrain.ini \
  make-db \
  -m ./marker_seeds/marker_seed_index.tsv \
  -r ./ncbi-genomes/index.tsv \
  -b kleb_ex \
  -bd ./blast-db \
  --min_pct_idty 90 \
  -o ./chronostrain-db/kleb_db.json
```
The argument `--min_pct_idty 90` reflects the belief (e.g. for MetaPhlAn markers) that within the species, there is
at least 90% similarity between any two instances (this is just an example; the value "90%" was arbitrarily chosen. 
It might not actually hold! Check this against BLAST before deciding on a cutoff value.)

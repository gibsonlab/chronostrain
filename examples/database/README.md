# 1. Summary

This contains an example of a database generation pipeline; there are multiple ways to 
arrive at the end product.
None of these are "core" to the package; thus we have provided a family of scripts instead.
(These ought to be modified/customized based on your needs)


## Try it out!

If you'd rather learn just by trying it out, check out the scripts located in the `complete_recipes` subdirectory.
For example, this script reproduces what was done in the paper, using the ingredients outlined below.

```bash
bash complete_recipes/ecoli.sh
```

We will proceed using `klebsiella.sh` (a much simpler recipe) as an example.

## What's included?

We provide scripts that sets up a RefSeq index, which is a prerequisite for the 
command `chronostrain make-db`.
The steps outlined here implement the preliminary database construction for the analyses done in the paper.

However, the database isn't necessarily limited to this recipe.
For instance, if you have high-quality MAGs (e.g. maybe you deeply sequenced a subset of your samples) 
then perhaps those should be used instead of downloading a consortium of reference genomes.


# 2. Requirements
- The user must separately install the tool `ncbi-genome-download`, which can be obtained from the original
author's <a href="https://github.com/kblin/ncbi-genome-download">github repo</a> or through
<a href="https://anaconda.org/bioconda/ncbi-genome-download">conda</a>.
- Since we will be extracting marker seeds from MetaPhlan (we will use version 4 as an example), the user should
also install <a href="https://huttenhower.sph.harvard.edu/metaphlan/">MetaPhlAn</a> and know where the 
database files are located. 
  
  By default (as of Oct. 2022), according to the MetaPhlAn documentation, one runs `metaphlan --install`, which downloads/extracts a tarball into 
  `<PYTHON_LIBS>/site-packages/metaphlan/metaphlan_databases`.
  

# 3. The Recipe, Explained

## Step 1: Download and compile a RefSeq index
Run the script `download_ncbi.sh`:
```
export NUM_CORES=4
export TARGET_TAXA=Klebsiella
export NCBI_REFSEQ_DIR=./ref_genomes
bash download_ncbi.sh
```

The recipe outlined in this script is as follows:
1. Use `ncbi-genome-download` (using 4 threads) to download all complete chromosomal assemblies of Klebsiella from NCBI. (*Quirk: You may see some errors (e.g. `No entry for file ending in <..>`) but these can mostly be ignored.*)
2. Run the custom python script `index_refseq.py` to create a TSV-formatted index of fasta/gff files 
   (see the main README for details), meant to serve as an input to `chronostrain make-db`.
   
## Step 2: Generate marker sequence seeds

### Strategy
In this example, we will extract all marker sequences from MetaPhlAn (version 3 or later) that include the
`g__Klebsiella` taxonomic label.
These will serve as *marker seeds* for ChronoStrain's database.

```bash
python extract_metaphlan_markers.py \
  -t g__Klebsiella \
  -i <python_site_packages>/metaphlan/metaphlan_databases/mpa_vJan21_CHOCOPhlAnSGB_202103.pkl \
  -o ./marker_seeds/marker_seed_index.tsv
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
homology or even >K% sequence identity across distantly-related species in the sample.
Here, K is the % identity threshold to use for determining marker sequences:
`chronostrain make-db --min-pct-idty K`.
By default, K is 75 (allowing for considerable single-nucleotide or structural variants within a species), so if 75% match identity (with near 100% query coverage) hits appear through BLAST when querying 
the marker seed, then that clade ought to also be included.

## Step 3: Use marker sequence seeds to create ChronoStrain DB

First, one needs to compile the reference sequences in the index into a BLAST database:
```bash
export BLAST_DB_DIR=./blast_db
export INDEX_FILE=./ref_genomes/index.tsv
export BLAST_DB_NAME=kleb_db
bash create_blast_db.sh
```
which simply concatenates all FASTA records in the RefSeq index and invokes `makeblastdb` to create
a database (with a specified name).

Then, we invoke chronostrain:
```bash
mkdir chronostrain-db
chronostrain -c chronostrain.ini \
  make-db \
  -m ./marker_seeds/marker_seed_index.tsv \
  -r ./ncbi-genomes/index.tsv \
  -b kleb_db \
  -bd ./blast_db \
  --min_pct_idty 90 \
  -o ./kleb.json
```
The argument `--min_pct_idty 90` reflects the belief (e.g. for MetaPhlAn markers) that within the species, there is
at least 90% similarity between any two instances (this is just an example; the value "90%" was arbitrarily chosen. 
It might not actually hold! Check this against BLAST before deciding on a cutoff value.)

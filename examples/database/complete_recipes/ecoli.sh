#### Ecoli marker definition using MLST Institute Pasteur + MetaPhlan4 markers.
# Since MLST genes are not "core" to E.coli, we will include all closely-aligning reference sequences
# from the entire Enterobacteriaceae family.

export NUM_CORES=8

export TARGET_TAXA="Biostraticola,Buttiauxella,Cedecea,Citrobacter,Cronobacter,Enterobacillus,Enterobacter,Escherichia,Franconibacter,Gibbsiella,Izhakiella,Klebsiella,Kluyvera,Kosakonia,Leclercia,Lelliottia,Limnobaculum,Mangrovibacter,Metakosakonia,Phytobacter,Pluralibacter,Pseudescherichia,Pseudocitrobacter,Raoultella,Rosenbergiella,Saccharobacter,Salmonella,Scandinavium,Shigella,Shimwellia,Siccibacter,Trabulsiella,Yokenella"
export METAPHLAN_TAXONOMIC_KEY='s__Escherichia_coli'
export TARGET_DIR=/mnt/e/ecoli_db

export NCBI_REFSEQ_DIR=${TARGET_DIR}/ref_genomes
export REFSEQ_INDEX=${TARGET_DIR}/ref_genomes/index.tsv
export BLAST_DB_DIR=${TARGET_DIR}/blast_db
export MARKER_SEED_INDEX=${TARGET_DIR}/marker_seeds/marker_seed_index.tsv
export CHRONOSTRAIN_TARGET_JSON=${TARGET_DIR}/ecoli.json

export BLAST_DB_NAME=Enterobacteriaceae_refseq
#export METAPHLAN_DB_PATH=~/miniconda3/envs/metaphlan/lib/python3.10/site-packages/metaphlan/metaphlan_databases/mpa_vJan21_CHOCOPhlAnSGB_202103.pkl
export METAPHLAN_DB_PATH=/mnt/e/metaphlan_databases/mpa_vJan21_CHOCOPhlAnSGB_202103/mpa_vJan21_CHOCOPhlAnSGB_202103.pkl # MetaPhlan 3 or newer
export MIN_PCT_IDTY=75
export CHRONOSTRAIN_DB_DIR=${TARGET_DIR}/chronostrain_files


# ========= Main body
# For an explanation, refer to the README.

if [ ! -f $METAPHLAN_DB_PATH ]
then
  echo "File ${METAPHLAN_DB_PATH} not found."
  exit 1
fi

bash download_ncbi.sh
bash create_blast_db.sh
python extract_metaphlan_markers.py -t $METAPHLAN_TAXONOMIC_KEY -i $METAPHLAN_DB_PATH -o ${TARGET_DIR}/marker_seeds/metaphlan_seeds.tsv
python mlst_download.py -t "Escherichia coli" -w ${TARGET_DIR}/mlst_schema -o ${TARGET_DIR}/marker_seeds/mlst_seeds.tsv
cat ${TARGET_DIR}/marker_seeds/*.tsv > ${MARKER_SEED_INDEX}

chronostrain -c chronostrain.ini \
  make-db \
  -m $MARKER_SEED_INDEX \
  -r $REFSEQ_INDEX \
  -b $BLAST_DB_NAME -bd $BLAST_DB_DIR \
  --min-pct-idty $MIN_PCT_IDTY \
  --ident-threshold 0.002 \
  -o $CHRONOSTRAIN_TARGET_JSON

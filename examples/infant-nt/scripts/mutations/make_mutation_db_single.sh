#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh


#require_program poppunk
#require_program demix_check
#require_program themisto
#require_program mGEMS


mutation_rate=$1
db_nickname=$2
poppunk_thresh=$3
require_variable "mutation_rate" $mutation_rate
require_variable "db_nickname" $db_nickname
require_variable "poppunk_thresh" $poppunk_thresh

target_dir=${DATA_DIR}/database/mutated_dbs/${db_nickname}
echo "[*] Creating databases with infant isolate mutations with rate=${mutation_rate} (dest=${target_dir})"
mutated_infant_isolate_index=${target_dir}/infant_index.tsv  # TSV file specifying locations of mutated isolate FASTA records.

echo "[**] Mutating infant isolate genomes."
python ../helpers/infant_isolate_mutate.py \
  -i ${INFANT_ISOLATE_INDEX} \
  -o ${mutated_infant_isolate_index} \
  -m ${mutation_rate} \
  -d ${target_dir}/mutated_isolates \
  --seed 314159


# ============= run ChronoStrain databse construction.
echo "[**] Running Chronostrain DB Construction."

chronostrain_db_dir=${target_dir}/chronostrain
mkdir -p ${chronostrain_db_dir}

log_file=${chronostrain_db_dir}/database.log
CHRONOSTRAIN_TARGET_JSON=${chronostrain_db_dir}/efaecalis.json
CHRONOSTRAIN_TARGET_CLUSTERS=${chronostrain_db_dir}/efaecalis.clusters.txt
BLAST_DB_DIR=${target_dir}/"blast_db"
BLAST_DB_NAME="Efcs_Europe_ELMC"  # Blast DB to create.
MARKER_SEED_INDEX=/mnt/e/infant_nt/database/marker_seeds/marker_seed_index.tsv
EUROPEAN_ISOLATE_INDEX=/data/local/europe_efaecalis/index.tsv
REFSEQ_INDEX=/mnt/e/infant_nt/database/enterococcaceae_index.tsv
NUM_CORES=8  # number of cores to use (e.g. for blastn)
MIN_PCT_IDTY=75  # accept BLAST hits as markers above this threshold.

#mkdir -p "$BLAST_DB_DIR"
#env JAX_PLATFORM_NAME=cpu \
#    CHRONOSTRAIN_LOG_INI=${BASE_DIR}/files/logging.ini \
#    CHRONOSTRAIN_LOG_FILEPATH=${log_file} \
#    CHRONOSTRAIN_DB_DIR=${chronostrain_db_dir} \
#    CHRONOSTRAIN_DB_JSON=${CHRONOSTRAIN_TARGET_JSON} \
#    CHRONOSTRAIN_CACHE_DIR=${target_dir}/.cache \
#    chronostrain -c ${BASE_DIR}/files/chronostrain.ini \
#        make-db \
#        -m $MARKER_SEED_INDEX \
#        -r $EUROPEAN_ISOLATE_INDEX \
#        -r $REFSEQ_INDEX \
#        -r $mutated_infant_isolate_index \
#        -b $BLAST_DB_NAME \
#        -bd $BLAST_DB_DIR \
#        --min-pct-idty $MIN_PCT_IDTY \
#        -o $CHRONOSTRAIN_TARGET_JSON \
#        --threads $NUM_CORES
#
## Perform clustering
#thresh=0.998
#env JAX_PLATFORM_NAME=cpu \
#    CHRONOSTRAIN_LOG_INI=${BASE_DIR}/files/logging.ini \
#    CHRONOSTRAIN_LOG_FILEPATH=${log_file} \
#    CHRONOSTRAIN_DB_DIR=${chronostrain_db_dir} \
#    CHRONOSTRAIN_DB_JSON=${CHRONOSTRAIN_TARGET_JSON} \
#    CHRONOSTRAIN_CACHE_DIR=${target_dir}/.cache \
#    chronostrain -c ${BASE_DIR}/files/chronostrain.ini \
#      cluster-db \
#      -i $CHRONOSTRAIN_TARGET_JSON \
#      -o $CHRONOSTRAIN_TARGET_CLUSTERS \
#      --ident-threshold ${thresh}

echo ${CHRONOSTRAIN_TARGET_CLUSTERS}
n_chrono_infant_clusters=$(cat ${CHRONOSTRAIN_TARGET_CLUSTERS} | grep GCA | wc -l)
n_chrono_total_clusters=$(cat ${CHRONOSTRAIN_TARGET_CLUSTERS} | wc -l)
echo "[!] ChronoStrain has ${n_chrono_infant_clusters} infant isolate clusters out of ${n_chrono_total_clusters}"

# ============= run mGEMS database construction.
echo "[**] Running mGEMS DB Construction (poppunk thresh=${poppunk_thresh})."

mgems_ref_dir=${target_dir}/mgems
mkdir -p ${mgems_ref_dir}
#bash mgems/helpers/run_poppunk_efaecalis.sh "${mgems_ref_dir}" "${mutated_infant_isolate_index}" "${poppunk_thresh}" # ensure that PopPUNK is installed for this script.

poppunk_threshold_clusts=${mgems_ref_dir}/poppunk/threshold/threshold_clusters.csv
n_poppunk_infant_clusters=$(cat ${poppunk_threshold_clusts} | grep GCA | awk -F',' '{print $2}' | uniq -c | wc -l)
n_poppunk_total_clusters=$(cat ${poppunk_threshold_clusts} | awk -F',' '{print $2}' | uniq -c | wc -l)
echo "[!] PopPUNK gave ${n_poppunk_infant_clusters} infant isolate clusters out of ${n_poppunk_total_clusters}"

python mgems/helpers/setup_efaecalis_index.py ${mgems_ref_dir}
demix_check --mode_setup --ref ${mgems_ref_dir} --threads ${N_CORES}

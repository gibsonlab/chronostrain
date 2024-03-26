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
require_variable "mutation_rate" $mutation_rate
require_variable "db_nickname" $db_nickname

target_dir=${DATA_DIR}/database/mutated_dbs/${db_nickname}
echo "Creating databases with infant isolate mutations with rate=${mutation_rate} (dest=${target_dir})"
mutated_infant_isolate_index=${target_dir}/infant_index.tsv  # TSV file specifying locations of mutated isolate FASTA records.

#python ../helpers/infant_isolate_mutate.py \
#  -i ${INFANT_ISOLATE_INDEX} \
#  -o ${mutated_infant_isolate_index} \
#  -m ${mutation_rate} \
#  -d ${target_dir}/mutated_isolates \
#  --seed 314159


# ============= run mGEMS database construction.
#mgems_ref_dir=${target_dir}/mgems
#mkdir -p ${mgems_ref_dir}
#bash mgems/helpers/run_poppunk_efaecalis.sh "${mgems_ref_dir}" "${mutated_infant_isolate_index}"  # ensure that PopPUNK is installed for this script.
#python mgems/helpers/setup_efaecalis_index.py ${mgems_ref_dir}
#demix_check --mode_setup --ref ${mgems_ref_dir} --threads ${N_CORES}


# ============= run ChronoStrain databse construction.
chronostrain_db_dir=${target_dir}/chronostrain
mkdir -p ${chronostrain_db_dir}

log_file=${chronostrain_db_dir}/database.log
CHRONOSTRAIN_TARGET_JSON=${chronostrain_db_dir}/efaecalis.json
BLAST_DB_DIR=${target_dir}/"blast_db"
BLAST_DB_NAME="Efcs_Europe_ELMC"  # Blast DB to create.
MARKER_SEED_INDEX=/mnt/e/infant_nt/database/marker_seeds/marker_seed_index.tsv
EUROPEAN_ISOLATE_INDEX=/data/local/europe_efaecalis/index.tsv
REFSEQ_INDEX=/mnt/e/infant_nt/database/enterococcaceae_index.tsv
NUM_CORES=8  # number of cores to use (e.g. for blastn)
MIN_PCT_IDTY=75  # accept BLAST hits as markers above this threshold.

mkdir -p "$BLAST_DB_DIR"
env JAX_PLATFORM_NAME=cpu \
    CHRONOSTRAIN_LOG_FILEPATH=${log_file} \
    CHRONOSTRAIN_DB_DIR=${chronostrain_db_dir} \
    CHRONOSTRAIN_DB_JSON=${CHRONOSTRAIN_TARGET_JSON} \
    CHRONOSTRAIN_CACHE_DIR=${target_dir}/.cache \
    CHRONOSTRAIN_LOG_INI=${BASE_DIR}/files/logging.ini \
    chronostrain -c ${BASE_DIR}/files/chronostrain.ini \
        make-db \
        -m $MARKER_SEED_INDEX \
        -r $EUROPEAN_ISOLATE_INDEX \
        -r $REFSEQ_INDEX \
        -r $mutated_infant_isolate_index \
        -b $BLAST_DB_NAME \
        -bd $BLAST_DB_DIR \
        --min-pct-idty $MIN_PCT_IDTY \
        -o $CHRONOSTRAIN_TARGET_JSON \
        --threads $NUM_CORES

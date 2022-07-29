# Modify these before running.
export PROJECT_DIR="/home/lactis/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export DATA_DIR="/mnt/e/umb_data"
export OUTPUT_DIR="/mnt/e/chronostrain/umb_stool"
export CHRONOSTRAIN_DB_DIR="/mnt/e/chronostrain/umb_database"
export SAMPLES_DIR="${DATA_DIR}/samples_stool"
export LOGDIR="${OUTPUT_DIR}/logs"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/stool/settings.sh"

# Location for reads and outputs.
export READS_DIR="${OUTPUT_DIR}/reads"
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/output/chronostrain"
export STRAINGE_OUTPUT_DIR="${OUTPUT_DIR}/output/strainge"

# sratools prefetch/sample output directories.
export SRA_CSV_PATH="${BASE_DIR}/files/umb_samples.csv"
export SRA_PREFETCH_DIR="${DATA_DIR}/prefetch"
export FASTERQ_TMP_DIR="${DATA_DIR}/fasterq-tmp"

# ========= Kneaddata
export KNEADDATA_DB_DIR="/mnt/e/kneaddata_db"
export NEXTERA_ADAPTER_PATH="/home/lactis/anaconda3/envs/chronostrain/share/trimmomatic/adapters/NexteraPE-PE.fa"
export TRIMMOMATIC_DIR="/home/lactis/anaconda3/envs/chronostrain/share/trimmomatic-0.39-2"

# ========= Chronostrain settings
export CHRONOSTRAIN_CORR_MODE='full'
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_NUM_ITERS=50
export CHRONOSTRAIN_NUM_SAMPLES=200
export CHRONOSTRAIN_READ_BATCH_SZ=2500
export CHRONOSTRAIN_LR_PATIENCE=5
export CHRONOSTRAIN_MIN_LR=1e-5
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_CACHE_DIR="${OUTPUT_DIR}/cache"
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

export CHRONOSTRAIN_DB_JSON_ALL="${CHRONOSTRAIN_DB_DIR}/database_all.json"
export CHRONOSTRAIN_DB_JSON_PRUNED="${CHRONOSTRAIN_DB_DIR}/database_pruned.json"
export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_JSON_PRUNED}"
export MULTIFASTA_FILE="strain_markers.fasta"

# ======== STRAINGE
export STRAINGE_DB=/mnt/e/strainge/straingst_umb/pan-genome-db.hdf5

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"

# ========= Done.
echo "======================================================="
echo "Loaded shell settings from ${SETTINGS_PATH}."
echo "Chronostrain config: ${CHRONOSTRAIN_INI}"
echo "Logging config: ${CHRONOSTRAIN_LOG_INI}"
echo "Logging dir: ${LOGDIR}"
echo "======================================================="

check_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}
export check_program

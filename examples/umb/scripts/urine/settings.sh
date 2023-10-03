# Modify these before running.
export PROJECT_DIR="/home/youn/work/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export DATA_DIR="/mnt/e/umb_data"
export OUTPUT_DIR="/data/cctm/youn/umb/urine"
export SAMPLES_DIR="${DATA_DIR}/samples_urine"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/urine/settings.sh"

# Location for reads and outputs.
export READS_DIR="${OUTPUT_DIR}/reads"

# sratools prefetch/sample output directories.
export SRA_PREFETCH_DIR="${DATA_DIR}/prefetch"
export FASTERQ_TMP_DIR="${DATA_DIR}/fasterq-tmp"

# ========= Kneaddata
export KNEADDATA_DB_DIR="/mnt/e/kneaddata_db"
export NEXTERA_ADAPTER_PATH="/home/lactis/anaconda3/envs/chronostrain/share/trimmomatic/adapters/NexteraPE-PE.fa"
export TRIMMOMATIC_DIR="/home/lactis/anaconda3/envs/chronostrain/share/trimmomatic-0.39-2"

# ========= Chronostrain settings
export CHRONOSTRAIN_CORR_MODE='full'
export CHRONOSTRAIN_NUM_ITERS=100
export CHRONOSTRAIN_NUM_SAMPLES=100
export CHRONOSTRAIN_READ_BATCH_SZ=10000
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_DECAY_LR=0.1
export CHRONOSTRAIN_LR=0.0005
export CHRONOSTRAIN_LOSS_TOL=1e-7
export CHRONOSTRAIN_LR_PATIENCE=5
export CHRONOSTRAIN_MIN_LR=1e-7
export CHRONOSTRAIN_CACHE_DIR="${OUTPUT_DIR}/cache"

export CHRONOSTRAIN_DB_JSON="/mnt/e/ecoli_db/ecoli.json"
export CHRONOSTRAIN_DB_DIR="/mnt/e/ecoli_db/chronostrain_files"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"

# ========= Done.
echo "======================================================="
echo "Loaded shell settings from ${SETTINGS_PATH}."
echo "Chronostrain config: ${CHRONOSTRAIN_INI}"
echo "Logging config: ${CHRONOSTRAIN_LOG_INI}"
echo "======================================================="

require_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}
export require_program

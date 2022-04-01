# Modify these before running.
export PROJECT_DIR="/home/bromii/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export DATA_DIR="/mnt/e/umb_data"
export OUTPUT_DIR="/mnt/d/chronostrain/umb_plate_scrapes"
export CHRONOSTRAIN_DB_DIR="/mnt/d/chronostrain/umb_database"
export LOGDIR="${OUTPUT_DIR}/logs"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/plate_scrapes/settings.sh"

# Location for reads and outputs.
export READS_DIR="${OUTPUT_DIR}/reads"
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/chronostrain"

# ========= Chronostrain settings
export CHRONOSTRAIN_NUM_ITERS=50
export CHRONOSTRAIN_NUM_SAMPLES=200
export CHRONOSTRAIN_READ_BATCH_SZ=5000
export CHRONOSTRAIN_NUM_EPOCHS=150
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.05
export CHRONOSTRAIN_CACHE_DIR="${OUTPUT_DIR}/cache"
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

export CHRONOSTRAIN_DB_JSON_ALL="${CHRONOSTRAIN_DB_DIR}/database_all.json"
export CHRONOSTRAIN_DB_JSON_PRUNED="${CHRONOSTRAIN_DB_DIR}/database_pruned.json"
export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_JSON_PRUNED}"
export MULTIFASTA_FILE="strain_markers.fasta"

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
# Modify these before running.
export PROJECT_DIR="/home/lactis/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export DATA_DIR="/mnt/e/umb_data"
export OUTPUT_DIR="/mnt/e/chronostrain/umb_plate_scrapes/output"
export OUTPUT_DIR="/mnt/e/chronostrain/umb_plate_scrapes/output"
export READS_DIR="/mnt/e/chronostrain/umb_plate_scrapes/reads"
export CHRONOSTRAIN_DB_DIR="/mnt/e/chronostrain/umb_database"
export LOGDIR="${OUTPUT_DIR}/logs"
export SAMPLES_DIR="/mnt/e/umb_data/plate_scrapes"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/plate_scrapes/settings.sh"

# Location for reads and outputs.
export READS_DIR="${OUTPUT_DIR}/reads"
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/chronostrain"

# ========= Chronostrain settings
export CHRONOSTRAIN_CORR_MODE='full'
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_NUM_ITERS=50
export CHRONOSTRAIN_NUM_SAMPLES=200
export CHRONOSTRAIN_READ_BATCH_SZ=5000
export CHRONOSTRAIN_DECAY_LR=0.5
export CHRONOSTRAIN_LR_PATIENCE=5
export CHRONOSTRAIN_MIN_LR=1e-5
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_CACHE_DIR="${OUTPUT_DIR}/cache"
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_DIR}/database_pruned_resolved.json"

# ================ assembly settings
export SPADES_DIR="/home/lactis/SPAdes-3.15.5"
export CHRONOSTRAIN_MARKERS_DIR=${CHRONOSTRAIN_DB_DIR}/__database_all_MARKERS
export MARKER_FASTA=all_markers.fasta
export BLAST_DB_NAME="markers"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"

# ========= Done.
if [ ! -z ${_settings_loaded} ]; then
	echo "======================================================="
	echo "Loaded shell settings from ${SETTINGS_PATH}."
	echo "Chronostrain config: ${CHRONOSTRAIN_INI}"
	echo "Logging config: ${CHRONOSTRAIN_LOG_INI}"
	echo "Logging dir: ${LOGDIR}"
	echo "======================================================="
fi
_settings_loaded="True"

require_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}
export require_program
# Modify these before running.
export PROJECT_DIR="/home/youn/work/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export OUTPUT_DIR="/mnt/e/umb_analysis/plate_scrapes"
export SAMPLES_DIR="/mnt/e/umb_data/plate_scrapes"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/plate_scrapes/settings.sh"

# Location for reads and outputs.
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/chronostrain"

# ========= Chronostrain settings
export CHRONOSTRAIN_CORR_MODE='strain'
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
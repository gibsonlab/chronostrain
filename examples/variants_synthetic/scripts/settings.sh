# Modify these before running.
export PROJECT_DIR="/mnt/f/microbiome_tracking"
export CHRONOSTRAIN_DATA_DIR="/mnt/f/microbiome_tracking/data/variants_synthetic"
export BASE_DIR="${PROJECT_DIR}/examples/variants_synthetic"
export N_CORES=4

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/settings.sh"

# Location for reads and outputs.
export OUTPUT_DIR="${CHRONOSTRAIN_DATA_DIR}/output"
export READS_DIR="${CHRONOSTRAIN_DATA_DIR}/reads"
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/chronostrain"

# The chronostrain input index.
export INPUT_INDEX_FILENAME="inputs.csv"

# ========= Chronostrain settings
export CHRONOSTRAIN_NUM_ITERS=2000
export CHRONOSTRAIN_NUM_SAMPLES=150
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"
export CHRONOSTRAIN_DB_DIR="${CHRONOSTRAIN_DATA_DIR}/database"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export LOGDIR="${CHRONOSTRAIN_DATA_DIR}/logs"
export TRUE_ABUNDANCE_PATH="${BASE_DIR}/files/true_abundances.csv"

# ========= ART specification
export READ_PROFILE_PATH="${BASE_DIR}/files/HiSeqReference"

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
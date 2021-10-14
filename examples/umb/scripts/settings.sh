# Modify these before running.
export PROJECT_DIR="/PHShome/yk847/chronostrain"
export CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/settings.sh"

# Location for reads and outputs.
export OUTPUT_DIR="${CHRONOSTRAIN_DATA_DIR}/umb/output"
export READS_DIR="${CHRONOSTRAIN_DATA_DIR}/umb/reads"
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/chronostrain"

# The chronostrain input index.
export INPUT_INDEX_PATH="${READS_DIR}/inputs.csv"

# sratools prefetch/sample output directories.
export SRA_PREFETCH_DIR="${CHRONOSTRAIN_DATA_DIR}/umb/prefetch"
export SAMPLES_DIR="${CHRONOSTRAIN_DATA_DIR}/umb/samples"

# ========= Bioproject/ncbi targets.
export BIOPROJECT="PRJNA400628"

# ========= Chronostrain settings
export CHRONOSTRAIN_NUM_ITERS=2000
export CHRONOSTRAIN_NUM_SAMPLES=150
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"
export CHRONOSTRAIN_DB_DIR="${CHRONOSTRAIN_DATA_DIR}/umb/database"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"

# ========= Done.
echo "Loaded shell settings from ${SETTINGS_PATH}."

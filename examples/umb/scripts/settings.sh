# Modify these before running.
export PROJECT_DIR="/mnt/d/microbiome_tracking"
export CHRONOSTRAIN_DATA_DIR="/home/younhun/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/single-run/settings.sh"

# Location for reads and outputs.
export OUTPUT_DIR="${CHRONOSTRAIN_DATA_DIR}/umb_output/"
export READS_DIR="${OUTPUT_DIR}/simulated_reads"
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/chronostrain"
export STRAINGE_OUTPUT_DIR="${OUTPUT_DIR}/strainge"

# ========= Chronostrain settings
export CHRONOSTRAIN_METHOD=bbvi
export CHRONOSTRAIN_NUM_ITERS=2000
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"
export CHRONOSTRAIN_DB_DIR="${CHRONOSTRAIN_DATA_DIR}/data"
export CHRONOSTRAIN_MARKERS_PATH="${CHRONOSTRAIN_DB_DIR}/marker_multifasta.fa"

# ========= StrainGE settings
export STRAINGE_DB_DIR="${CHRONOSTRAIN_DATA_DIR}/strainge_db"
export STRAINGE_DB_PATH="${STRAINGE_DB_DIR}/ecoli-db.hdf5"
export STRAINGE_OUTPUT_FILENAME="abundances.csv"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export TRUE_ABUNDANCE_PATH="${BASE_DIR}/files/true_abundances.csv"

# ========= Done.
echo "Loaded shell settings from ${SETTINGS_PATH}."

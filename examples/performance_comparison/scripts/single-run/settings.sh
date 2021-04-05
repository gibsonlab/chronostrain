# Modify these before running.
export PROJECT_DIR="/mnt/f/microbiome_tracking"
export CHRONOSTRAIN_DATA_DIR="/home/younhun/chronostrain"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/single-run/settings.sh"

# ======== Read settings (how many to sample/read length/where they are stored)
export N_READS=10000000
export READ_LEN=150

# Location for reads and outputs.
TRIAL_DIR="${CHRONOSTRAIN_DATA_DIR}/runs/reads_${N_READS}/single-run"
export READS_DIR="${TRIAL_DIR}/simulated_reads"
export CHRONOSTRAIN_OUTPUT_DIR="${TRIAL_DIR}/output/chronostrain"
export METAPHLAN_OUTPUT_DIR="${TRIAL_DIR}/output/metaphlan"

# ========= Chronostrain settings
export CHRONOSTRAIN_METHOD=em
export CHRONOSTRAIN_NUM_ITERS=5000
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

# ========= MetaPhlAn settings
export METAPHLAN_DB="${CHRONOSTRAIN_DATA_DIR}/metaphlan_db"
export METAPHLAN_DB_INDEX="mpa_v30_CHOCOPhlAn_201901"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export BASE_DIR="${PROJECT_DIR}/examples/performance_comparison"
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export TRUE_ABUNDANCE_PATH="${BASE_DIR}/files/true_abundances.csv"
export READ_PROFILE_PATH="${BASE_DIR}/files/HiSeqReference"

# ========= Done.
echo "Loaded shell settings from ${SETTINGS_PATH}."

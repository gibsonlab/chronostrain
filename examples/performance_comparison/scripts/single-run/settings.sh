# Modify these before running.
export PROJECT_DIR="/mnt/f/microbiome_tracking"
export CHRONOSTRAIN_DATA_DIR="/home/younhun/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/performance_comparison"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/single-run/settings.sh"

# ======== Read settings (how many to sample/read length/where they are stored)
export N_READS=1000000
export READ_LEN=150
#export QUALITY_SHIFT="-3"
export QUALITY_SHIFT="0"

# Location for reads and outputs.
export TRIAL_DIR="${CHRONOSTRAIN_DATA_DIR}/runs/reads_${N_READS}/qs_${QUALITY_SHIFT}/single-run"
export READS_DIR="${TRIAL_DIR}/simulated_reads"
export CHRONOSTRAIN_OUTPUT_DIR="${TRIAL_DIR}/output/chronostrain"
export METAPHLAN_OUTPUT_DIR="${TRIAL_DIR}/output/metaphlan"
export STRAINGE_OUTPUT_DIR="${TRIAL_DIR}/output/strainge"

# ========= Chronostrain settings
export CHRONOSTRAIN_METHOD=bbvi
export CHRONOSTRAIN_NUM_ITERS=2000
#export CHRONOSTRAIN_METHOD=em
#export CHRONOSTRAIN_NUM_ITERS=5000
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"
export CHRONOSTRAIN_DB_DIR="${CHRONOSTRAIN_DATA_DIR}/data"
export CHRONOSTRAIN_MARKERS_PATH="${CHRONOSTRAIN_DB_DIR}/marker_multifasta.fa"

# ========= MetaPhlAn settings
export METAPHLAN_DB="${CHRONOSTRAIN_DATA_DIR}/metaphlan_db"
export METAPHLAN_DB_INDEX="mpa_v30_CHOCOPhlAn_201901"

# ========= StrainGE settings
export STRAINGE_DB_DIR="${CHRONOSTRAIN_DATA_DIR}/strainge_db"
export STRAINGE_DB_PATH="${STRAINGE_DB_DIR}/ecoli-db.hdf5"
export STRAINGE_OUTPUT_FILENAME="abundances.csv"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export TRUE_ABUNDANCE_PATH="${BASE_DIR}/files/true_abundances.csv"
export READ_PROFILE_PATH="${BASE_DIR}/files/HiSeqReference"

# ========= Done.
echo "Loaded shell settings from ${SETTINGS_PATH}."

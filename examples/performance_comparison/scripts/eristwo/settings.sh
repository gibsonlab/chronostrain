# Modify these before running.
export PROJECT_DIR="/PHShome/yk847/chronostrain"
export CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"
export CONDA_ENV="chronostrain"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/eristwo/settings.sh"

# ======== Read settings (how many to sample/read length/where they are stored)
export N_READS_MIN=1000000
export N_READS_MAX=10000000
export N_READS_STEP=1000000
export N_TRIALS=10
export READ_LEN=150

# Location for reads and outputs.
export RUNS_DIR="${CHRONOSTRAIN_DATA_DIR}/runs"

# ======== LSF settings
# (note: 1000 = 1gb)
export LSF_READGEN_QUEUE="normal"
export LSF_READGEN_MEM=10000
export LSF_READGEN_N_CORES=1

export LSF_CHRONOSTRAIN_QUEUE="normal"
export LSF_CHRONOSTRAIN_MEM=40000
export LSF_CHRONOSTRAIN_N_CORES=1

export LSF_METAPHLAN_QUEUE="normal"
export LSF_METAPHLAN_MEM=10000
export LSF_METAPHLAN_N_CORES=4

# Turn this to `1` if you want the scripts to automatically submit the LSF jobs.
export LSF_AUTO_SUBMIT=0

# LSF files and stdout/stderr output locations.
export LSF_DIR="${CHRONOSTRAIN_DATA_DIR}/lsf_files"
export READGEN_LSF_DIR="${LSF_DIR}/readgen"
export READGEN_LSF_OUTPUT_DIR="${READGEN_LSF_DIR}/output"
export CHRONOSTRAIN_LSF_DIR="${LSF_DIR}/chronostrain"
export CHRONOSTRAIN_LSF_OUTPUT_DIR="${CHRONOSTRAIN_LSF_DIR}/output"
export METAPHLAN_LSF_DIR="${LSF_DIR}/metaphlan"
export METAPHLAN_LSF_OUTPUT_DIR="${METAPHLAN_LSF_DIR}/output"

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

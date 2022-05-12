# Modify these before running.
export PROJECT_DIR="/home/bromii/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/synthetic"
export DATA_DIR="/mnt/e/synthetic_data"
export CHRONOSTRAIN_DB_DIR="/mnt/d/chronostrain/synthetic/database"

# ==================== Read sampling settings
export N_TRIALS=20
export READ_LEN=150
export N_CORES=4

# Ground truth.
export GROUND_TRUTH="${BASE_DIR}/files/ground_truth.csv"

# ART specification
export READ_PROFILE_PATH="${BASE_DIR}/files/HiSeqReference"

# ================= StrainGST settings
export STRAINGST_DB_DIR=${CHRONOSTRAIN_DB_DIR}/straingst
export STRAINGST_DB_HDF5=${STRAINGST_DB_DIR}/database.hdf5

# ================= StrainEst settings
export STRAINEST_DB_DIR=${CHRONOSTRAIN_DB_DIR}/strainest
export STRAINEST_BOWTIE2_DB_NAME='bt2_strains'

# ================= ConStrains settings
export CONSTRAINS_DIR=/home/bromii/constrains
export METAPHLAN2_DIR=/home/bromii/MetaPhlAn2

# ========= Chronostrain settings
export INFERENCE_SEED=31415
export CHRONOSTRAIN_NUM_ITERS=50
export CHRONOSTRAIN_NUM_SAMPLES=200
export CHRONOSTRAIN_READ_BATCH_SZ=2500
export CHRONOSTRAIN_NUM_EPOCHS=150
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.01
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_DB_JSON="${BASE_DIR}/files/database.json"
export MULTIFASTA_FILE="markers.fasta"
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"


check_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}


get_trial_dir()
{
	n_reads=$1
	trial=$2
	trial_dir="${DATA_DIR}/reads_${n_reads}/trial_${trial}"
	echo ${trial_dir}
}

export check_program

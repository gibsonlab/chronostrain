# Modify these before running.
export PROJECT_DIR="/home/youn/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/infant-nt"
export DATA_DIR="/mnt/e/infant_nt"

# Dataset
export DATASET_METAGENOMIC_CATALOG="${DATA_DIR}/project.tsv"
export NT_RESULT_TABLE="${BASE_DIR}/files/babybiome_lineages_by_time_point.tsv"

# ========= Chronostrain settings
export CHRONOSTRAIN_NUM_ITERS=100
export CHRONOSTRAIN_NUM_SAMPLES=200
export CHRONOSTRAIN_READ_BATCH_SZ=10000
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_LOSS_TOL=1e-5
export CHRONOSTRAIN_LR_PATIENCE=5
export CHRONOSTRAIN_MIN_LR=1e-6
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"


# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"

export CHRONOSTRAIN_DB_DIR="${DATA_DIR}/database/chronostrain_files"
export CHRONOSTRAIN_DB_JSON="${DATA_DIR}/database/efaecalis.json"
export CHRONOSTRAIN_DB_NAME="efaecalis"

export TRIMMOMATIC_PATH=/home/youn/miniconda3/envs/chronostrain/share/trimmomatic-0.39-2


require_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}

require_variable()
{
	var_name=$1
	value=$2
	if [ -z "$value" ]
	then
		echo "Environment variable \"$var_name\" is empty"
		exit 1
	fi
}

require_file()
{
	path=$1
	if [ ! -f $path ]
	then
		echo "File ${path} not found."
		exit 1
	fi
}

require_dir()
{
	path=$1
	if [ ! -d $path ]
	then
		echo "Directory ${path} not found."
		exit 1
	fi
}

export require_program
export require_variable
export require_file
export require_dir

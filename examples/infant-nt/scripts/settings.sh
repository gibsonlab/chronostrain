# Modify these before running.
export PROJECT_DIR="/home/youn/work/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/infant-nt"
export DATA_DIR="/mnt/e/infant_nt"

# Dataset
export DATASET_METAGENOMIC_CATALOG="${DATA_DIR}/project.tsv"
export ENA_ISOLATE_ASSEMBLY_CATALOG=${DATA_DIR}/isolate_assembly_ena.tsv
export NT_RESULT_TABLE="${BASE_DIR}/files/babybiome_lineages_by_time_point.tsv"


# ========= Chronostrain settings
export CHRONOSTRAIN_NUM_ITERS=100
export CHRONOSTRAIN_NUM_SAMPLES=100
export CHRONOSTRAIN_READ_BATCH_SZ=10000
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_DECAY_LR=0.1
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_LOSS_TOL=1e-5
export CHRONOSTRAIN_LR_PATIENCE=5
export CHRONOSTRAIN_MIN_LR=1e-6


# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export CHRONOSTRAIN_DB_DIR="${DATA_DIR}/database/chronostrain_files"
export CHRONOSTRAIN_DB_JSON="${DATA_DIR}/database/chronostrain_files/efaecalis.json"
export CHRONOSTRAIN_DB_NAME="efaecalis"

export TRIMMOMATIC_PATH=/home/youn/mambaforge/envs/chronostrain/share/trimmomatic-0.39-2


# ========== Posthoc analysis
export REFSEQ_INDEX=${DATA_DIR}/database/ref_genomes/index.tsv

export MARKER_SEED_INDEX=${DATA_DIR}/database/marker_seeds/marker_seed_index.tsv
export MARKER_BLAST_DB_DIR=${DATA_DIR}/database/posthoc/marker_blast_db
export MARKER_BLAST_DB_NAME="marker_seeds"
export MARKER_ALIGNMENT_DIR=${DATA_DIR}/database/_ALIGN_efaecalis/multiple_alignment


# ========= Utility functions.
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

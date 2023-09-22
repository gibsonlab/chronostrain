# Modify these before running.
export PROJECT_DIR="/home/youn/work/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/semisynthetic"
export DATA_DIR="/mnt/e/semisynthetic_data"

export MISC_DB_DIR="${DATA_DIR}/databases"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/cache"  # Default for all scripts; specify per script if desired!

# ==================== Read sampling settings
export N_GENOME_REPLICATES=10
export N_TRIALS=2
export MUTATION_RATIOS=("1.0")
export SYNTHETIC_COVERAGES=(2500 5000 10000 20000 40000)
export READ_LEN=150
export N_CORES=12
export BASE_GENOME_MUTATION_RATE=0.002
export NOISE_GENOME_MUTATION_RATE=0.002

export RELATIVE_GROUND_TRUTH="${BASE_DIR}/files/ground_truth.csv"
export BACKGROUND_CSV="${BASE_DIR}/files/background.csv"
export BACKGROUND_FASTQ_DIR="${DATA_DIR}/background"
export SRA_PREFETCH_DIR="${BACKGROUND_FASTQ_DIR}/prefetch"
export FASTERQ_TMP_DIR="${BACKGROUND_FASTQ_DIR}/fasterq-tmp"

# ========= Kneaddata
export KNEADDATA_DB_DIR="/mnt/e/kneaddata_db"
export NEXTERA_ADAPTER_PATH="/home/youn/mambaforge/envs/chronostrain/share/trimmomatic/adapters/NexteraPE-PE.fa"
export TRIMMOMATIC_DIR="/home/youn/mambaforge/envs/chronostrain/share/trimmomatic-0.39-2"

# ================= StrainFacts+gt-pro settings
export CALLM_BIN_DIR=/home/youn/CallM
export KMC_BIN_DIR=/home/youn/KMC/bin
export GT_PRO_BIN_DIR=/home/youn/gt-pro
export GT_PRO_DB_DIR=/mnt/e/gt-pro_db
export GT_PRO_DB_NAME=ecoli_db

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export REFSEQ_INDEX="/mnt/e/ecoli_db/ref_genomes/index.tsv"


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


get_replicate_dir()
{
  mutation_ratio=$1
  replicate=$2
	echo "${DATA_DIR}/mutratio_${mutation_ratio}/replicate_${replicate}"
}


get_trial_dir()
{
  mutation_ratio=$1
  replicate=$2
  read_depth=$3
  trial=$4
	rep_dir=$(get_replicate_dir "$mutation_ratio" "$replicate")
	echo "${rep_dir}/reads_${read_depth}/trial_${trial}"
}

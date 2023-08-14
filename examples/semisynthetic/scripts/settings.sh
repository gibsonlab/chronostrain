# Modify these before running.
export PROJECT_DIR="/home/youn/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/semisynthetic"
export DATA_DIR="/mnt/e/semisynthetic_data"
export UMB_DB_DIR="/mnt/e/chronostrain/umb_database"

export MISC_DB_DIR="${DATA_DIR}/databases"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/cache"  # Default for all scripts; specify per script if desired!
export CHRONOSTRAIN_DB_DIR="${MISC_DB_DIR}/chronostrain"

# ==================== Read sampling settings
export N_TRIALS=10
export READ_LEN=150
export N_CORES=8
export BACKGROUND_N_READS=1000000

export RELATIVE_GROUND_TRUTH="${BASE_DIR}/files/ground_truth.csv"
export BACKGROUND_CSV="${BASE_DIR}/files/background.csv"
export BACKGROUND_FASTQ_DIR="${DATA_DIR}/background"
export SRA_PREFETCH_DIR="${BACKGROUND_FASTQ_DIR}/prefetch"
export FASTERQ_TMP_DIR="${BACKGROUND_FASTQ_DIR}/fasterq-tmp"

# ART specification
export READ_PROFILE_PATH="${BASE_DIR}/files/HiSeqReference"

# ========= Kneaddata
export KNEADDATA_DB_DIR="/mnt/e/kneaddata_db"
export NEXTERA_ADAPTER_PATH="/home/youn/anaconda3/envs/chronostrain/share/trimmomatic/adapters/NexteraPE-PE.fa"
export TRIMMOMATIC_DIR="/home/youn/anaconda3/envs/chronostrain/share/trimmomatic-0.39-2"

# ================= StrainGST settings
export STRAINGST_DB_DIR=${MISC_DB_DIR}/straingst
export STRAINGST_CHROMOSOME_DB_HDF5=${STRAINGST_DB_DIR}/chromosome_db.hdf5
export STRAINGST_MARKER_DB_HDF5=${STRAINGST_DB_DIR}/marker_db.hdf5

# ================= StrainEst settings
export STRAIN_REP_FASTA=/mnt/e/ref_genomes/human_readable/refseq/bacteria/Escherichia/coli/K-12_MG1655/NZ_CP010438.1.chrom.fna
export STRAINEST_DB_DIR=${MISC_DB_DIR}/strainest
export STRAINEST_BT2_DB=ecoli_db
export SYNTHETIC_COVERAGES=(2500 5000 10000 25000 50000)

# ================= StrainFacts+gt-pro settings
export CALLM_BIN_DIR=/home/youn/CallM
export KMC_BIN_DIR=/home/youn/KMC/bin
export GT_PRO_BIN_DIR=/home/youn/gt-pro
export GT_PRO_DB_DIR=/mnt/e/gt-pro_db
export GT_PRO_DB_NAME=ecoli_db

# ========= Chronostrain settings
export CHRONOSTRAIN_NUM_ITERS=100
export CHRONOSTRAIN_NUM_SAMPLES=100
export CHRONOSTRAIN_READ_BATCH_SZ=10000
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_LOSS_TOL=1e-5
export CHRONOSTRAIN_LR_PATIENCE=5
export CHRONOSTRAIN_MIN_LR=1e-6
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export REFSEQ_INDEX="/mnt/e/ref_genomes/index.tsv"
export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_DIR}/all_strains.json"
export ECOLI_ONLY_JSON="${CHRONOSTRAIN_DB_DIR}/ecoli_only.json"
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"


require_program()
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

export require_program

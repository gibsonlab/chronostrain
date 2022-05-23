# Modify these before running.
export PROJECT_DIR="/home/bromii/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/semisynthetic"
export DATA_DIR="/mnt/e/semisynthetic_data"
export CHRONOSTRAIN_DB_DIR="/mnt/d/chronostrain/umb_database"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/cache"  # Default for all scripts; specify per script if desired!
export MISC_DB_DIR="/mnt/d/chronostrain/semisynthetic"

# ==================== Read sampling settings
export N_TRIALS=20
export READ_LEN=150
export N_CORES=4
export BACKGROUND_N_READS=1000000

export RELATIVE_GROUND_TRUTH="${BASE_DIR}/files/ground_truth.csv"
export BACKGROUND_SRA_ID="SRR061229"
export BACKGROUND_FASTQ_DIR="${DATA_DIR}/background"
export BACKGROUND_FQ_1="${BACKGROUND_FASTQ_DIR}/${BACKGROUND_SRA_ID}_1.fastq"
export BACKGROUND_FQ_2="${BACKGROUND_FASTQ_DIR}/${BACKGROUND_SRA_ID}_2.fastq"
export SRA_PREFETCH_DIR="${DATA_DIR}/prefetch"
export FASTERQ_TMP_DIR="${DATA_DIR}/fasterq-tmp"

# ART specification
export READ_PROFILE_PATH="${BASE_DIR}/files/HiSeqReference"

# ================= StrainGST settings
export STRAINGST_DB_DIR=${MISC_DB_DIR}/straingst
export STRAINGST_CHROMOSOME_DB_HDF5=${STRAINGST_DB_DIR}/chromosome_db.hdf5
export STRAINGST_MARKER_DB_HDF5=${STRAINGST_DB_DIR}/marker_db.hdf5

# ================= StrainEst settings
export STRAIN_REP_FASTA=/mnt/e/ref_genomes/human_readable/refseq/bacteria/Escherichia/coli/K-12_MG1655/NZ_CP010438.1.chrom.fna
export STRAINEST_DB_DIR=${MISC_DB_DIR}/strainest
export STRAINEST_BT2_DB=ecoli_db

# ========= Chronostrain settings
export INFERENCE_SEED=31415
export CHRONOSTRAIN_NUM_ITERS=50
export CHRONOSTRAIN_NUM_SAMPLES=200
export CHRONOSTRAIN_READ_BATCH_SZ=2500
export CHRONOSTRAIN_NUM_EPOCHS=150
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.05
export CHRONOSTRAIN_LR_PATIENCE=10
export CHRONOSTRAIN_MIN_LR=1e-4
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
#export CHRONOSTRAIN_DB_JSON="${BASE_DIR}/files/database.json"
export REFSEQ_INDEX="/mnt/e/ref_genomes/index.tsv"
export CHRONOSTRAIN_DB_JSON="${DATA_DIR}/database.json"
export MULTIFASTA_FILE="ecoli_only.fasta"
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

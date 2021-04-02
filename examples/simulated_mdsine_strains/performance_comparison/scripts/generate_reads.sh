#!/bin/bash
set -e

if [ -z ${PROJECT_DIR} ]; then
	echo "Variable 'PROJECT_DIR' is not set. Exiting."
	exit 1
else
	echo "PROJECT_DIR=${PROJECT_DIR}"
fi

# ======================================
N_READS_MIN=10000
N_READS_MAX=100000
N_READS_STEP=10000
N_TRIALS=15

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs/read_sample.log"

TRUE_ABUNDANCE_PATH="${BASE_DIR}/true_abundances.csv"

RUNS_DIR="${BASE_DIR}/runs"
READ_LEN=150

LSF_QUEUE="big"
CONDA_ENV="chronostrain"
LSF_MEM=10000
LSF_DIR="${BASE_DIR}/lsf_files"
LSF_OUTPUT_DIR="${LSF_DIR}/output"
# =====================================
# The location of the ReadGenAndFiltering repo for sampling
READGEN_DIR="/PHShome/yk847/Read-Generation-and-Filtering"
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH
mkdir -p $LSF_DIR
mkdir -p $LSF_OUTPUT_DIR

# =========================================================
# Database initialization. (pre-download fasta and markers.)
python $BASE_DIR/scripts/initialize_database.py
# =========================================================

for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
    echo "[Number of reads: ${n_reads}, trial #${trial}]"
		LSF_PATH="${LSF_DIR}/sample_reads_${n_reads}_trial_${trial}.lsf"

    TRIAL_DIR="${RUNS_DIR}/trials/reads_${n_reads}_trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
    mkdir -p $READS_DIR
    SEED=$trial

		cat <<- EOFDOC > $LSF_PATH
		#!/bin/bash
		#BSUB -J read_sample
		#BSUB -o ${LSF_OUTPUT_DIR}/read_sample_${n_reads}_${trial}-%J.out
		#BSUB -e ${LSF_OUTPUT_DIR}/read_sample_${n_reads}_${trial}-%J.err
		#BSUB -q $LSF_QUEUE
		#BSUB -n 1
		#BSUB -M ${LSF_MEM}
		#BSUB -R rusage[mem=${LSF_MEM}]

		python ${PROJECT_DIR}/scripts/readgen.py \
		$n_reads \
		$READ_LEN \
		$trial \
		$READGEN_DIR/profiles/HiSeqReference \
		$READGEN_DIR/profiles/HiSeqReference \
		$TRUE_ABUNDANCE_PATH \
		$READS_DIR \
		$SEED
		EOFDOC
  done
done

# ============== Submit all LSF jobs. ================
for lsf_file in ${LSF_OUTPUT_DIR}/*.lsf
do
	bsub < $lsf_file
done

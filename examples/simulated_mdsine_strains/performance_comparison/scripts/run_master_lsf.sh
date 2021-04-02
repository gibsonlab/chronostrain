#!/bin/bash
set -e

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/PHShome/yk847/chronostrain"
# =====================================

LSF_QUEUE="gpu"
CONDA_ENV="chronostrain"
# (note: 1000 = 1gb)
LSF_CHRONOSTRAIN_MEM=10000
LSF_CHRONOSTRAIN_N_CORES=1
LSF_METAPHLAN_MEM=10000
LSF_METAPHLAN_N_CORES=4

N_READS_MIN=10000
N_READS_MAX=100000
N_READS_STEP=10000
N_TRIALS=15

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"
LSF_DIR="${BASE_DIR}/lsf_files"
LSF_OUTPUT_DIR="${LSF_DIR}/output"
# =====================================

export PROJECT_DIR
mkdir -p $LSF_DIR
mkdir -p $LSF_OUTPUT_DIR

# ================ LSF creation ===========================
for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		CHRONOSTRAIN_LSF_PATH="${LSF_DIR}/reads_${n_reads}_trial_${trial}-chronostrain.lsf"
		METAPHLAN_LSF_PATH="${LSF_DIR}/reads_${n_reads}_trial_${trial}-metaphlan.lsf"

		# Generate LSF files via heredoc.
		# ============ Chronostrain LSF ============
		echo "Creating ${CHRONOSTRAIN_LSF_PATH}"
		cat <<- EOFDOC > $CHRONOSTRAIN_LSF_PATH
		#!/bin/bash
		#BSUB -J bench_chronostrain
		#BSUB -o ${LSF_OUTPUT_DIR}/chronostrain_${n_reads}_${trial}-%J.out
		#BSUB -e ${LSF_OUTPUT_DIR}/chronostrain_${n_reads}_${trial}-%J.err
		#BSUB -q $LSF_QUEUE
		#BSUB -n ${LSF_CHRONOSTRAIN_N_CORES}
		#BSUB -M ${LSF_CHRONOSTRAIN_MEM}
		#BSUB -R rusage[mem=${LSF_CHRONOSTRAIN_MEM}]

		source activate ${CONDA_ENV}
		export PROJECT_DIR=${PROJECT_DIR}
		bash run_chronostrain.sh ${n_reads} ${trial}
		EOFDOC

		# ============ Metaphlan LSF ============
		echo "Creating ${METAPHLAN_LSF_PATH}"
		cat <<- EOFDOC > $METAPHLAN_LSF_PATH
		#!/bin/bash
		#BSUB -J bench_metaphlan
		#BSUB -o ${LSF_OUTPUT_DIR}/metaphlan_${n_reads}_${trial}-%J.out
		#BSUB -e ${LSF_OUTPUT_DIR}/metaphlan_${n_reads}_${trial}-%J.err
		#BSUB -q $LSF_QUEUE
		#BSUB -n ${LSF_METAPHLAN_N_CORES}
		#BSUB -M ${LSF_METAPHLAN_MEM}
		#BSUB -R rusage[mem=${LSF_METAPHLAN_MEM}]

		source activate ${CONDA_ENV}
		export PROJECT_DIR=${PROJECT_DIR}
		bash run_metaphlan.sh ${n_reads} ${trial}
		EOFDOC
	done
done

# ============== Submit all LSF jobs. ================
for lsf_file in ${LSF_OUTPUT_DIR}/*.lsf
do
	bsub < $lsf_file
done

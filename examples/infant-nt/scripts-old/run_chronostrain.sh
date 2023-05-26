#!/bin/bash
set -e

source settings.sh

participant=$1

if [ -z "$participant" ]
then
	echo "var \"participant\" is empty"
	exit 1
fi

# =========== Run chronostrain. ==================
echo "Running inference on participant ${participant}."

run_dir=${DATA_DIR}/${participant}/chronostrain
export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/inference.log"
export CHRONOSTRAIN_CACHE_DIR="${run_dir}/.cache"
#export CHRONOSTRAIN_DB_SPECIFICATION="${DATA_DIR}/${participant}/isolate_assemblies/metadata.tsv"
#export CHRONOSTRAIN_DB_NAME="P-${participant}"
#export CHRONOSTRAIN_DB_JSON="${DATA_DIR}/${participant}/${participant}_chronostrain.json"
#export CHRONOSTRAIN_PKL_JSON="${CHRONOSTRAIN_DB_DIR}/mlst.pkl"
#export CHRONOSTRAIN_PKL_JSON="${CHRONOSTRAIN_DB_DIR}/__mlst_pruned_MARKERS/database.posix.pkl"
export CHRONOSTRAIN_DB_NAME="mlst_pruned"
cd ${BASE_DIR}


if ! [ -f ${run_dir}/filtered/FILTER_DONE.txt ]
then
  chronostrain filter \
    -r ${run_dir}/reads.csv \
    -o ${run_dir}/filtered \
    --aligner "bwa-mem2"
  touch ${run_dir}/filtered/FILTER_DONE.txt
else
  echo "[*] Skipping filter (already done!)"
fi

#env CUDA_VISIBLE_DEVICES='' chronostrain advi \
chronostrain advi \
  -r ${run_dir}/filtered/filtered_reads.csv \
  -o ${run_dir}/inference \
  --correlation-mode "strain" \
  --iters $CHRONOSTRAIN_NUM_ITERS \
  --epochs $CHRONOSTRAIN_NUM_EPOCHS \
  --decay-lr $CHRONOSTRAIN_DECAY_LR \
  --lr-patience ${CHRONOSTRAIN_LR_PATIENCE} \
	--loss-tol ${CHRONOSTRAIN_LOSS_TOL} \
  --learning-rate $CHRONOSTRAIN_LR \
  --num-samples $CHRONOSTRAIN_NUM_SAMPLES \
  --no-allocate-fragments \
  --read-batch-size $CHRONOSTRAIN_READ_BATCH_SZ \
	--min-lr ${CHRONOSTRAIN_MIN_LR} \
  --plot-format "pdf" \
  --plot-elbo \
	--prune-strains \
	--with-zeros

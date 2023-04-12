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
export CHRONOSTRAIN_DB_JSON="${DATA_DIR}/${participant}/${participant}_chronostrain.json"
cd ${BASE_DIR}


if ! [ -f ${run_dir}/filtered/FILTER_DONE.txt ]
then
  chronostrain filter \
    -r ${run_dir}/reads.csv \
    -o ${run_dir}/filtered \
    --aligner "bowtie2"
  touch ${run_dir}/filtered/FILTER_DONE.txt
else
  echo "[*] Skipping filter (already done!)"
fi

chronostrain advi \
  -r ${run_dir}/filtered/filtered_reads.csv \
  -o ${run_dir}/inference \
  --seed $INFERENCE_SEED \
  --correlation-mode "time" \
  --iters $CHRONOSTRAIN_NUM_ITERS \
  --epochs $CHRONOSTRAIN_NUM_EPOCHS \
  --decay-lr $CHRONOSTRAIN_DECAY_LR \
  --lr-patience ${CHRONOSTRAIN_LR_PATIENCE} \
  --min-lr ${CHRONOSTRAIN_MIN_LR} \
  --learning-rate $CHRONOSTRAIN_LR \
  --num-samples $CHRONOSTRAIN_NUM_SAMPLES \
  --no-allocate-fragments \
  --read-batch-size $CHRONOSTRAIN_READ_BATCH_SZ \
  --plot-format "pdf" \
  --plot-elbo

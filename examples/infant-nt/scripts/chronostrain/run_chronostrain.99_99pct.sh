#!/bin/bash
set -e
source settings.sh

participant=$1
require_variable 'participant' $participant
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
echo "[*] Running inference on participant ${participant}."

run_dir=${DATA_DIR}/${participant}/chronostrain_99_99pct
export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/inference.log"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/.cache"
cd ${BASE_DIR}


chronostrain advi \
  -r ${run_dir}/filtered/filtered_reads.csv \
  -o ${run_dir}/inference \
  -s ${DATA_DIR}/database/chronostrain_files/efaecalis.clusters_99_99pct.txt \
  --correlation-mode "full" \
  --iters ${CHRONOSTRAIN_NUM_ITERS} \
  --epochs ${CHRONOSTRAIN_NUM_EPOCHS} \
  --decay-lr ${CHRONOSTRAIN_DECAY_LR} \
  --lr-patience ${CHRONOSTRAIN_LR_PATIENCE} \
  --loss-tol ${CHRONOSTRAIN_LOSS_TOL} \
  --learning-rate ${CHRONOSTRAIN_LR} \
  --num-samples ${CHRONOSTRAIN_NUM_SAMPLES} \
  --read-batch-size ${CHRONOSTRAIN_READ_BATCH_SZ} \
  --min-lr ${CHRONOSTRAIN_MIN_LR} \
  --plot-elbo \
  --prune-strains \
  --with-zeros \
  --prior-p 0.001 \
  --accumulate-gradients

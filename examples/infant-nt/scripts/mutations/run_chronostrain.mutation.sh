#!/bin/bash
set -e
source settings.sh
source chronostrain/settings.sh

participant=$1
mutation_rate=$2
require_variable 'participant' $participant
require_variable 'mutation_rate' $mutation_rate
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
echo "[*] Running inference on participant ${participant}, with mutated db p=0.${mutation_rate}."

run_dir=${DATA_DIR}/${participant}/chronostrain_mutation_${mutation_rate}
cd ${BASE_DIR}


export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/inference.log"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/.cache_mut"
export CHRONOSTRAIN_DB_JSON=${DATA_DIR}/database/mutated_dbs/${mutation_rate}/chronostrain/efaecalis.json
export CHRONOSTRAIN_CLUSTERS=${DATA_DIR}/database/mutated_dbs/${mutation_rate}/chronostrain/efaecalis.clusters.txt

chronostrain advi \
  -r ${run_dir}/filtered/filtered_reads.csv \
  -o ${run_dir}/inference \
  -s ${CHRONOSTRAIN_CLUSTERS} \
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
#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

export CHRONOSTRAIN_LOG_FILEPATH="/mnt/e/caesarian_data/513122/chronostrain/inference.log"
export CHRONOSTRAIN_CACHE_DIR="/mnt/e/caesarian_data/513122/chronostrain/.cache"
cd ${BASE_DIR}


#python helpers/filter.py \
#	-r /mnt/e/caesarian_data/513122/chronostrain/reads.csv \
#	-o /mnt/e/caesarian_data/513122/chronostrain/filtered \
#  --db-pickle /home/youn/.chronostrain/databases/mlst.pkl \
#	--aligner "bowtie2"


python helpers/inference.py \
  -r /mnt/e/caesarian_data/513122/chronostrain/filtered/filtered_reads.csv \
  -o /mnt/e/caesarian_data/513122/chronostrain \
  --seed $INFERENCE_SEED \
  --db-pickle /home/youn/.chronostrain/databases/mlst.pkl \
  --correlation-mode "time" \
  --iters $CHRONOSTRAIN_NUM_ITERS \
  --epochs $CHRONOSTRAIN_NUM_EPOCHS \
  --decay-lr $CHRONOSTRAIN_DECAY_LR \
  --lr-patience ${CHRONOSTRAIN_LR_PATIENCE} \
  --min-lr ${CHRONOSTRAIN_MIN_LR} \
  --learning-rate $CHRONOSTRAIN_LR \
  --num-samples $CHRONOSTRAIN_NUM_SAMPLES \
  --read-batch-size $CHRONOSTRAIN_READ_BATCH_SZ \
  --plot-format "pdf" \
  --plot-elbo

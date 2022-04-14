#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

for umb_id in UMB05 UMB08 UMB11 UMB12 UMB15 UMB18 UMB20 UMB23 UMB24
do
    echo "Running inference on ${umb_id}."
    export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/chronostrain_${umb_id}.log"

    cd $PROJECT_DIR/scripts
    python run_bbvi.py \
    --reads_input "${READS_DIR}/${umb_id}_filtered/filtered_${umb_id}_inputs.csv" \
    --out_dir $CHRONOSTRAIN_OUTPUT_DIR/${umb_id} \
    --seed $SEED \
    --iters $CHRONOSTRAIN_NUM_ITERS \
    --epochs $CHRONOSTRAIN_NUM_EPOCHS \
    --decay_lr $CHRONOSTRAIN_DECAY_LR \
    --lr_patience 5 \
    --min_lr 1e-4 \
    --learning_rate $CHRONOSTRAIN_LR \
    --num_samples $CHRONOSTRAIN_NUM_SAMPLES \
    --read_batch_size $CHRONOSTRAIN_READ_BATCH_SZ \
    --plot_format "pdf" \
    --plot_elbo
done
# ================================================
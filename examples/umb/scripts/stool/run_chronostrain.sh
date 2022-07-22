#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

#for umb_id in UMB05 UMB08 UMB11 UMB12 UMB15 UMB18 UMB20 UMB23 UMB24
for umb_id in UMB01 UMB02 UMB03 UMB04 UMB06 UMB07 UMB09 UMB10 UMB13 UMB14 UMB16 UMB17 UMB19 UMB21 UMB22 UMB25 UMB26 UMB27 UMB28 UMB29 UMB30 UMB31
do
    echo "Running inference on ${umb_id}."
    export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/chronostrain_${umb_id}.log"
		export CHRONOSTRAIN_CACHE_DIR="$CHRONOSTRAIN_OUTPUT_DIR/${umb_id}/cache"

    cd $PROJECT_DIR/scripts
    python run_advi.py \
    --reads_input "${READS_DIR}/${umb_id}_filtered/filtered_${umb_id}_inputs.csv" \
    --out_dir $CHRONOSTRAIN_OUTPUT_DIR/${umb_id} \
    --seed $SEED \
    --correlation_mode $CHRONOSTRAIN_CORR_MODE \
    --iters $CHRONOSTRAIN_NUM_ITERS \
    --epochs $CHRONOSTRAIN_NUM_EPOCHS \
    --decay_lr $CHRONOSTRAIN_DECAY_LR \
    --lr_patience $CHRONOSTRAIN_LR_PATIENCE \
    --min_lr $CHRONOSTRAIN_MIN_LR \
    --learning_rate $CHRONOSTRAIN_LR \
    --num_samples $CHRONOSTRAIN_NUM_SAMPLES \
    --read_batch_size $CHRONOSTRAIN_READ_BATCH_SZ \
    --plot_format "pdf" \
    --plot_elbo
done
# ================================================
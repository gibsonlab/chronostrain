#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

for umb_id in UMB01 UMB02 UMB03 UMB04 UMB05 UMB06 UMB07 UMB08 UMB09 UMB10 UMB11 UMB12 UMB13 UMB14 UMB15 UMB16 UMB17 UMB18 UMB19 UMB20 UMB21 UMB22 UMB23 UMB24 UMB25 UMB26 UMB27 UMB28 UMB29 UMB30 UMB31
do
    echo "Running inference on ${umb_id}."
    export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/chronostrain_${umb_id}.log"

    chronostrain advi \
		-r "${READS_DIR}/${umb_id}_filtered/filtered_${umb_id}_inputs.csv" \
		-o $CHRONOSTRAIN_OUTPUT_DIR/${umb_id} \
		--seed $SEED \
    --correlation_mode $CHRONOSTRAIN_CORR_MODE \
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
done
# ================================================

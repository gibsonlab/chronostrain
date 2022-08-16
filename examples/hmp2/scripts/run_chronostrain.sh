#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

while IFS=, read -r patient
do
    echo "Running inference on ${patient}."
    index_filename="filtered_${INPUT_INDEX_FILENAME}"
    export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/chronostrain_${patient}.log"

    cd $PROJECT_DIR/scripts
    chronostrain advi \
           --reads "${READS_DIR}/${patient}/filtered/${index_filename}" \
           --out-dir $CHRONOSTRAIN_OUTPUT_DIR/${patient} \
           --seed $SEED \
           --iters $CHRONOSTRAIN_NUM_ITERS \
           --epochs $CHRONOSTRAIN_NUM_EPOCHS \
           --decay-lr $CHRONOSTRAIN_DECAY_LR \
           --lr-patience 10 \
           --min-lr 1e-4 \
           --learning-rate $CHRONOSTRAIN_LR \
           --num-samples $CHRONOSTRAIN_NUM_SAMPLES \
           --read-batch-size $CHRONOSTRAIN_BATCH_SZ \
           --plot-format "pdf" \
           --plot-elbo
done < ${BASE_DIR}/files/patients.txt
# ================================================
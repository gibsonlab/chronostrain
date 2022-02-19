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
    python run_bbvi.py \
           --reads_input "${READS_DIR}/${patient}/filtered/${index_filename}" \
           --out_dir $CHRONOSTRAIN_OUTPUT_DIR/${patient} \
           --seed $SEED \
           --iters $CHRONOSTRAIN_NUM_ITERS \
           --epochs $CHRONOSTRAIN_NUM_EPOCHS \
           --decay_lr $CHRONOSTRAIN_DECAY_LR \
           --lr_patience 10 \
           --min_lr 1e-4 \
           --learning_rate $CHRONOSTRAIN_LR \
           --num_samples $CHRONOSTRAIN_NUM_SAMPLES \
           --frag_chunk_size $CHRONOSTRAIN_FRAG_CHUNK_SZ \
           --plot_format "pdf" \
           --plot_elbo
done < ${BASE_DIR}/files/patients.txt
# ================================================
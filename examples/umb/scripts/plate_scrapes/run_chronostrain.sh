#!/bin/bash
set -e

source settings.sh

# =========== Run chronostrain. ==================
cd $READS_DIR
for csv_file in *.csv
do
	regex="(.*).csv"
	if [[ $csv_file =~ $regex ]]
	then
		sample_name="${BASH_REMATCH[1]}"
	else
		echo "Skipping ${csv_file}."
		continue
	fi

	echo "Running inference on ${sample_name}."
	export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/chronostrain_${sample_name}.log"

	chronostrain advi \
		-r "${READS_DIR}/${sample_name}_filtered/filtered_${sample_name}.csv" \
		-o $CHRONOSTRAIN_OUTPUT_DIR/${sample_name} \
		--seed $SEED \
    --correlation-mode $CHRONOSTRAIN_CORR_MODE \
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
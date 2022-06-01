#!/bin/bash
set -e

source settings.sh
SEED=31415

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

	cd $PROJECT_DIR/scripts
	python run_advi.py \
	--reads_input "${READS_DIR}/${sample_name}_filtered/filtered_${sample_name}.csv" \
	--out_dir $CHRONOSTRAIN_OUTPUT_DIR/${sample_name} \
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
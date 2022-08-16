#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Read filtering. ===============
cd $READS_DIR
for csv_file in *.csv
do
	regex="(.*).csv"
	if [[ $csv_file =~ $regex ]]
	then
		sample_name="${BASH_REMATCH[1]}"
	else
		echo "Skipping."
		continue
	fi

	export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter_${sample_name}.log"

	echo "Filtering reads for ${sample_name}"
	chronostrain filter \
	-r ${READS_DIR}/${csv_file} \
	-o "${READS_DIR}/${sample_name}_filtered"
done

#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"
SEED=31415

# =========== Read filtering. ===============

for umb_id in UMB01 UMB02 UMB03 UMB04 UMB05 UMB06 UMB07 UMB08 UMB09 UMB10 UMB11 UMB12 UMB13 UMB14 UMB15 UMB16 UMB17 UMB18 UMB19 UMB20 UMB21 UMB22 UMB23 UMB24 UMB25 UMB26 UMB27 UMB28 UMB29 UMB30 UMB31
do
	export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter_${umb_id}.log"

	echo "Filtering reads for ${umb_id}"
	chronostrain filter \
	-r "${READS_DIR}/${umb_id}_inputs.csv" \
	-o "${READS_DIR}/${umb_id}_filtered"
done

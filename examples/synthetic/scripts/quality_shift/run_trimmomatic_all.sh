#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts/quality_shift
for (( q_shift = ${Q_SHIFT_MIN}; q_shift < ${Q_SHIFT_MAX}+1; q_shift += ${Q_SHIFT_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		trial_dir=$(get_trial_dir $q_shift $trial)
		bash trimmomatic.sh ${trial_dir}
	done
done
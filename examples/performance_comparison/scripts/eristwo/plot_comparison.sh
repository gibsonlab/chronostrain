#!/bin/bash
set -e

source settings.sh

# Index path
> $OUTPUT_INDEX_PATH

echo "Generating result index: ${OUTPUT_INDEX_PATH}"
for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( quality_shift = ${Q_SHIFT_MIN}; quality_shift < ${Q_SHIFT_MAX}+1; quality_shift += ${Q_SHIFT_STEP} ));
	do
		for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
		do
			# =============== Trial-specific settings ===================
			CHRONOSTRAIN_LSF_PATH="${CHRONOSTRAIN_LSF_DIR}/reads_${n_reads}_qs_${quality_shift}_trial_${trial}.lsf"
			STRAINGE_LSF_PATH="${STRAINGE_LSF_DIR}/reads_${n_reads}_qs_${quality_shift}_trial_${trial}.lsf"

			TRIAL_DIR="${RUNS_DIR}/reads_${n_reads}/qs_${quality_shift}/trial_${trial}"
			READS_DIR="${TRIAL_DIR}/simulated_reads"
			CHRONOSTRAIN_OUTPUT_DIR="${TRIAL_DIR}/output/chronostrain"
			STRAINGE_OUTPUT_DIR="${TRIAL_DIR}/output/strainge"

			echo "Chronostrain,${trial},${quality_shift},${CHRONOSTRAIN_OUTPUT_DIR}/samples.pt" >> $OUTPUT_INDEX_PATH
			echo "StrainGE (Markers/Filtered),${trial},${quality_shift},${STRAINGE_OUTPUT_DIR}/markers_filtered/abundances.csv" >> $OUTPUT_INDEX_PATH
			echo "StrainGE (Markers/Unfiltered),${trial},${quality_shift},${STRAINGE_OUTPUT_DIR}/markers_unfiltered/abundances.csv" >> $OUTPUT_INDEX_PATH
			echo "StrainGE (Genome),${trial},${quality_shift},${STRAINGE_OUTPUT_DIR}/genomes/abundances.csv" >> $OUTPUT_INDEX_PATH
			echo "StrainGE (Full),${trial},${quality_shift},${STRAINGE_OUTPUT_DIR}/full/abundances.csv" >> $OUTPUT_INDEX_PATH
		done
	done
done

echo "Generating plot: ${PERFORMANCE_PLOT_PATH}"
python ${BASE_DIR}/scripts/helpers/plot_performances.py \
-t $OUTPUT_INDEX_PATH \
-g $TRUE_ABUNDANCE_PATH \
-o $PERFORMANCE_PLOT_PATH \
--title "Performance Comparison" \
--font_size 22 \
--thickness 2 \
--draw_legend \
--format "pdf"

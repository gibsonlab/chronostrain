#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/better_plot.log"
SEED=31415

# =========== Run chronostrain. ==================
echo "Drawping plots."

for umb_id in UMB05 UMB08 UMB11 UMB12 UMB15 UMB18 UMB20 UMB23 UMB24
do
	cd $PROJECT_DIR/scripts
	python plot_bbvi_samples.py \
	--reads_input "${READS_DIR}/${umb_id}_filtered/${index_filename}" \
	--out_path $CHRONOSTRAIN_OUTPUT_DIR/${umb_id}/better_plot.pdf \
	--samples_path $CHRONOSTRAIN_OUTPUT_DIR/${umb_id}/samples.pt \
	--title "${umb_id}" \
	--draw_legend \
	--plot_format "PDF" \
	--width 8 \
	--height 6 \
	--dpi 30 \
	--strain_trunc_level 5e-3
done
# ================================================

#!/bin/bash
set -e

source settings_singlegene.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"


# Perform multiple alignment of marker genes and reads.
python ${BASE_DIR}/examples/umb/helpers/align_all.py \
-i "${READS_DIR}/*_filtered_singlegene/*.fastq" \
-o multi_align_out_path_TODO \
-m marker_name_TODO \
-w work_dir_TODO \
-t 4


# Run assembly.


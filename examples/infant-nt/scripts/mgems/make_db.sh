#!/bin/bash
set -e

thresh=0.00036  # this matches chronostrain 99.8% sim threshold on infant isolates.
#thresh=0.000001  # this matches chronostrain 99.99% sim threshold

bash mgems/helpers/run_poppunk_efaecalis.sh ${EFAECALIS_CHRONO_MIRROR_REF_DIR} ${INFANT_ISOLATE_INDEX} ${thresh} # ensure that PopPUNK is installed for this script.
python mgems/helpers/setup_efaecalis_index.py ${EFAECALIS_CHRONO_MIRROR_REF_DIR}

# ====== demix_check python script
# this takes a very long time to run.
demix_check --mode_setup --ref ${EFAECALIS_CHRONO_MIRROR_REF_DIR} --threads ${N_CORES}

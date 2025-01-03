#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh


thresh=0.00036  # this matches chronostrain 99.8% sim threshold on infant isolates.
#thresh=0.000001  # this matches chronostrain 99.99% sim threshold


## Based on the guide from https://github.com/PROBIC/mSWEEP/blob/master/docs/pipeline.md
## create poppunk input
poppunk_outdir=${EFAECALIS_CHRONO_MIRROR_REF_DIR}/poppunk
mkdir -p ${poppunk_outdir}
cd "${poppunk_outdir}"


echo "[!] Preparing PopPUNK inputs. (${poppunk_outdir})"
sed '1d' "${EUROPE_EFAECALIS_INDEX}" | cut -f4,6 > poppunk_input.tsv
sed '1d' "${INFANT_ISOLATE_INDEX}" | cut -f4,6 >> poppunk_input.tsv


### Expected empirical result using this script: 808 clusters
# print this.
n_clusters=$(asdfasd)
echo "# clusters after poppunk: ${n_clusters}"

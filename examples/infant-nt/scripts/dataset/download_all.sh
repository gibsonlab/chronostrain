#!/bin/bash
set -e
source settings.sh
require_file $ENA_ISOLATE_ASSEMBLY_CATALOG
require_file $DATASET_METAGENOMIC_CATALOG


cd ${BASE_DIR}/scripts
while read -r infant_id; do
  echo "[*] Handling ${infant_id}"

  mkdir -p ${DATA_DIR}/${infant_id}/reads
  breadcrumb=${DATA_DIR}/${infant_id}/reads/download.DONE

  if [ -f ${breadcrumb} ]; then
    echo "Download already done for ${infant_id}"
  else
    bash dataset/download_dataset.sh ${infant_id}
  fi
done < "${INFANT_ID_LIST}"

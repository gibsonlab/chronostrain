#!/bin/bash
set -e
source settings.sh


# ====== Download and catalog assemblies from isolate sequencing project
if ! [ -f ${ENA_ISOLATE_ASSEMBLY_CATALOG} ]
then
  echo "[*] Downloading assembly catalog."
  curl -G "https://www.ebi.ac.uk/ena/portal/api/filereport" \
    -d 'result=assembly' \
    -d 'accession=PRJEB22252' \
    -d 'fields=accession,sample_accession,scientific_name,sample_title,assembly_name' \
    -d 'format=tsv' \
    -d 'limit=0' \
    -o ${ENA_ISOLATE_ASSEMBLY_CATALOG}
fi


cd ${BASE_DIR}/scripts
python ${BASE_DIR}/helpers/list_all_participants.py ${ENA_ISOLATE_ASSEMBLY_CATALOG} | while read line
do
  participant=$line
  process_mark=${DATA_DIR}/${participant}/chronostrain/process_reads.DONE

  if [[ -f $process_mark ]]; then
    echo "[*] Skipping processing of ${participant}"
  else
    echo "[*] Handling ${participant}"
    mkdir -p ${DATA_DIR}/${participant}/reads
    bash download_dataset.sh ${participant}
    if [[ -f ${DATA_DIR}/${participant}/dataset.tsv ]]; then
      bash ../helpers/process_dataset.sh ${participant}  # Run pre-processing on reads.
      touch $process_mark
    else
      echo "Didn't find valid reads for ${participant}."
    fi
  fi
done

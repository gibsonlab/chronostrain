#!/bin/bash
set -e
source settings.sh


while read line
do
  participant=$line
  echo $participant
  participant_dir=${DATA_DIR}/${participant}

  if [[ ! -d $participant_dir ]]; then
    continue
  fi

  if [[ ! -f "${participant_dir}/chronostrain/process_reads.DONE" ]]; then
    continue
  fi

  if [[ ! -f "${participant_dir}/dataset.tsv" ]]; then
    continue
  fi

  while IFS=$'\t' read participant timepoint sampleid read1 read2
  do
    if [ "${participant}" == "Participant" ]; then continue; fi
    echo "Handling participant ${participant}, sample ${sampleid}"
    bash strainge/run_straingst.sh ${participant} ${sampleid} ${timepoint}
  done < ${participant_dir}/dataset.tsv
done < "${INFANT_ID_LIST}"
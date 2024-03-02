#!/bin/bash
set -e
source settings.sh


while read line
do
  participant=$line
  bash mgems/run_pipeline_infant.sh "${participant}"
done < "${INFANT_ID_LIST}"

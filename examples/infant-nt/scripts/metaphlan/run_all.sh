#!/bin/bash
set -e
source settings.sh


while read line
do
  infant_id=$line
  bash metaphlan/run_infant.sh "${infant_id}"
done < "${INFANT_ID_LIST}"

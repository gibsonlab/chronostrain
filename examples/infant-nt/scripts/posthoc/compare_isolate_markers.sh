#!/bin/bash
set -e
source settings.sh


require_program 'blastn'
require_file $MARKER_SEED_INDEX
require_dir $MARKER_ALIGNMENT_DIR


echo "[*] Parsing BLAST results to compute distances."


perform_comparison()
{
  participant=$1
  assembly_accession=$2

  isolate_dir=${DATA_DIR}/${participant}/isolate_assemblies

  export CHRONOSTRAIN_CACHE_DIR=.
  export CHRONOSTRAIN_LOG_FILEPATH=${isolate_dir}/${assembly_accession}_alignments/run.log
  export CHRONOSTRAIN_DB_JSON="${DATA_DIR}/database/efaecalis-1raw.json"
  export CHRONOSTRAIN_DB_NAME="efaecalis-1raw"

  # Signal JAX that we won't need the GPU
  env JAX_PLATFORM_NAME=cpu python posthoc/compare_isolate_markers.py \
    --accession ${assembly_accession} \
    --isolate-dir ${isolate_dir} \
    --marker-seeds $MARKER_SEED_INDEX \
    --seed-multiple-alignments $MARKER_ALIGNMENT_DIR \
    --threads 8
}


shopt -s nullglob
python ${BASE_DIR}/helpers/list_all_participants.py ${ENA_ISOLATE_ASSEMBLY_CATALOG} | while read participant
do
  participant_dir=${DATA_DIR}/${participant}/isolate_assemblies

  for f in $participant_dir/*.fasta; do
    acc="$(basename $f .fasta)"

    output_file=${participant_dir}/${acc}_alignments/distances.tsv
    if [ -f $output_file ]; then
      echo "[**] Overwriting ${acc} distances for ${participant}"
      perform_comparison $participant $acc
    else
      echo "[**] Handling ${acc} distances for ${participant}"
      perform_comparison $participant $acc
    fi
  done
done

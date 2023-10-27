#!/bin/bash
set -e
source settings.sh

require_program 'blastn'
require_variable 'MARKER_BLAST_DB_DIR' $MARKER_BLAST_DB_DIR
require_variable 'MARKER_BLAST_DB_NAME' $MARKER_BLAST_DB_NAME


echo "[*] Running BLAST on isolate assemblies to identify markers."


blast_isolates_of()
{
  participant=$1
  isolate_dir=${DATA_DIR}/${participant}/isolate_assemblies

  if [ ! -d $isolate_dir ]
  then
    echo "ERROR: Isolate dir ${isolate_dir} not found. Exiting."
    exit 1
  fi

  blast_fasta_in_dir $isolate_dir $output_file
}


blast_fasta_in_dir()
{
  query_dir=$1
  output_file=$2

  _cwd=$(pwd)
  cd $query_dir
  for f in *.fasta; do
    query_basename=$(basename $f .fasta)   # usually is the isolate accession ID
    output_file="${query_basename}.marker_blast.tsv"
    echo "-> Querying ${query_basename}"
    run_blastn $f $output_file
  done
  cd $_cwd
}


run_blastn()
{
  query_file=$1
  output_file=$2
  # run blastn with args
  env BLASTDB=${MARKER_BLAST_DB_DIR} blastn \
    -db ${MARKER_BLAST_DB_NAME} \
    -query ${query_file} \
    -outfmt "6 saccver sstart send slen qseqid qstart qend qlen evalue pident gaps qcovhsp" \
    -out ${output_file} \
    -strand both
}


# Iterate through all infants.
python ${BASE_DIR}/helpers/list_all_participants.py ${ENA_ISOLATE_ASSEMBLY_CATALOG} | while read line
do
  participant=$line
  echo "[*] Handling BLAST queries for participant ${participant}"
  blast_isolates_of $participant
done

#!/bin/bash
set -e
source settings.sh

require_program 'makeblastdb'
require_variable 'MARKER_BLAST_DB_DIR' $MARKER_BLAST_DB_DIR
require_variable 'MARKER_BLAST_DB_NAME' $MARKER_BLAST_DB_NAME
require_file $MARKER_SEED_INDEX


# Create a blast DB of marker seeds.
echo "[*] Creating Blast database of marker seeds."

# Temporary FASTA file to feed into makeblastdb.
refseq_fasta=__tmp_markers.fasta
echo "Target fasta file: ${refseq_fasta}"

# Create/clear the file.
mkdir -p ${MARKER_BLAST_DB_DIR}
> ${MARKER_BLAST_DB_DIR}/${refseq_fasta}

# Concatenate marker seeds into multi-fasta file.
while IFS=$'\t' read marker_name fasta_path metadata
do
  echo "Concatenating ${marker_name}..."
  echo ">${marker_name}" >> ${MARKER_BLAST_DB_DIR}/${refseq_fasta}
  tail -n +2 ${fasta_path} >> ${MARKER_BLAST_DB_DIR}/${refseq_fasta}
done < $MARKER_SEED_INDEX


# Invoke makeblastdb.
cd ${MARKER_BLAST_DB_DIR}
makeblastdb \
  -in ${refseq_fasta} \
  -out $MARKER_BLAST_DB_NAME \
  -dbtype nucl \
  -title $MARKER_BLAST_DB_NAME \
  -parse_seqids

# Clean up.
rm $refseq_fasta
cd -

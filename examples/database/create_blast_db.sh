#!/bin/bash
set -e
source settings.sh

require_program 'makeblastdb'

require_variable 'BLAST_DB_DIR' $BLAST_DB_DIR
require_variable 'REFSEQ_INDEX' $REFSEQ_INDEX
require_variable 'BLAST_DB_NAME' $BLAST_DB_NAME


echo "[*] Creating Blast database."
refseq_fasta=__tmp_refseqs.fasta
echo "Target fasta file: ${refseq_fasta}"

mkdir -p ${BLAST_DB_DIR}

# Create a concatenated file of fasta records.
> ${BLAST_DB_DIR}/${refseq_fasta}  # Clear file
while IFS=$'\t' read genus species strain accession assembly seqpath chrlen gffpath
do
	if [ "${seqpath}" == "SeqPath" ]; then continue; fi
	echo "Concatenating ${seqpath}..."
	cat ${seqpath} >> ${BLAST_DB_DIR}/${refseq_fasta}
done < ${REFSEQ_INDEX}

# Invoke makeblastdb.
cd ${BLAST_DB_DIR}
makeblastdb \
-in ${refseq_fasta} \
-out $BLAST_DB_NAME \
-dbtype nucl \
-title $BLAST_DB_NAME \
-parse_seqids

# Clean up.
rm ${refseq_fasta}
cd -


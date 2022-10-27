#!/bin/bash
set -e
source settings.sh


echo "[*] Creating Blast database."
refseq_fasta=__tmp_refseqs.fasta
echo "Target fasta file: ${refseq_fasta}"

mkdir -p ${BLAST_DB_DIR}

# Create a concatenated file of fasta records.
> ${BLAST_DB_DIR}/${refseq_fasta}  # Clear file
while IFS=$'\t' read genus species strain accession assembly seqpath chrlen gffpath
do
	echo "Concatenating ${seqpath}..."
	cat ${seqpath} >> ${BLAST_DB_DIR}/${refseq_fasta}
done < ${INDEX_FILE}

# Invoke makeblastdb.
cd ${BLAST_DB_DIR}
makeblastdb \
-in ${refseq_fasta} \
-out kleb_ex \
-dbtype nucl \
-title "Example Klebsiella DB" \
-parse_seqids

# Clean up.
rm ${refseq_fasta}
cd -


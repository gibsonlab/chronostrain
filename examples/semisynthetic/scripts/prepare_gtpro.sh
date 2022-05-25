#!/bin/bash
set -e
source settings.sh

cd ${GT_PRO_DB_DIR}

# Make fasta directory, fill it with relevant genomes
fasta_dir=fastas
mkdir -p $fasta_dir
cd $fasta_dir
python ${BASE_DIR}/helpers/list_strain_paths.py -j ${CHRONOSTRAIN_DB_JSON} -i $REFSEQ_INDEX \
| while read strain_seq; do
	base_name="$(basename -- $strain_seq)"
	base_name="${base_name%.chrom.fna}"
	echo "[*] Linking $base_name (file: ${strain_seq})"
	ln -s $strain_seq $base_name.fna
done
cd ..

# Extract core snps.
out_dir=.
${CALLM_BIN} genomes --fna-dir ${fasta_dir} --out-dir ${out_dir}

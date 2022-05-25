#!/bin/bash
set -e
source settings.sh

cd ${GT_PRO_DB_DIR}

# Make fasta directory, fill it with relevant genomes
species_dir=ecoli
fasta_dir=${species_dir}/genomes
mkdir -p $fasta_dir
cd $fasta_dir

python ${BASE_DIR}/helpers/list_strain_paths.py -j ${CHRONOSTRAIN_DB_JSON} -i $REFSEQ_INDEX \
| while read strain_seq; do
	base_name="$(basename -- $strain_seq)"
	base_name="${base_name%.chrom.fna}"
	sym_link="${base_name}.fna"
	if [ -L ${sym_link} ]; then
		echo "[*] Symbolic link ${sym_link} already found."
	else
		echo "[*] Linking $base_name (file: ${strain_seq})"
		ln -s $strain_seq ${sym_link}
	fi
done
cd ${GT_PRO_DB_DIR}

# Extract core snps.
${CALLM_BIN} genomes --fna-dir ${fasta_dir} --out-dir ${species_dir} --threads ${N_CORES}
rm -rf ${species_dir}/temp

echo "./ecoli" > build.list
mv ecoli/tag_msa.fna ecoli/msa.fa
${GT_PRO_BIN} build --in ./build.list --out ./ecoli_db --dbname ecoli_db --threads ${N_CORES}

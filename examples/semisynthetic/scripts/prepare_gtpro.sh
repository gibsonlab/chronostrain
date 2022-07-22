#!/bin/bash
set -e
source settings.sh


export PATH=${PATH}:${KMC_BIN_DIR}:${CALLM_BIN_DIR}:${GT_PRO_BIN_DIR}
check_program kmc_dump
check_program CallM
check_program GT_Pro

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
CallM genomes --fna-dir ${fasta_dir} --out-dir ${species_dir} --threads ${N_CORES}
rm -rf ${species_dir}/temp

echo "./ecoli" > build.list
mv ecoli/tag_msa.fna ecoli/msa.fa
export PATH=$PATH:${KMC_BIN_DIR}

echo "[*] invoking GT_Pro build."
GT_Pro build --in ./build.list --out ${GT_PRO_DB_NAME} --dbname ${GT_PRO_DB_NAME} --threads ${N_CORES} --overwrite

# Remove the header line (otherwise GT_Pro parse breaks.)
tail -n +1 ${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME}.snp_dict.tsv > ${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME}.snp_dict.noheader.tsv

echo "[*] invoking GT_Pro optimize."
sample_fastq=${DATA_DIR}/reads_100000/trial_1/reads/0_reads_1.fq.gz
GT_Pro optimize -d ./${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME} -i $sample_fastq

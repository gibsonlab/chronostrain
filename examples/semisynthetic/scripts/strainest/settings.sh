# ================= Themisto + mSWEEP settings
export STRAIN_REP_FASTA=/mnt/e/ecoli_db/ref_genomes/human_readable/refseq/bacteria/Escherichia/coli/K-12_MG1655/NZ_CP010438.1.chrom.fna

export META_ALIGN_DB_DIR=${DATA_DIR}/StrainEst_files
export META_ALIGN_FASTA=${META_ALIGN_DB_DIR}/MR.fasta
export META_ALIGN_BOWTIE2_DB=${META_ALIGN_DB_DIR}/metagenome_alignment_db


get_strainest_db_dir()
{
  replicate=$1
  replicate_dir=$(get_replicate_dir "${replicate}")
	echo "${replicate_dir}/databases/StrainEst"
}

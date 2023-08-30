# ================= Themisto + mSWEEP settings
export THEMISTO_BIN_DIR=/home/youn/work/themisto/build/bin
export POPPUNK_REFSEQ_DIR=${DATA_DIR}/poppunk


get_themisto_db_dir()
{
  mutation_ratio=$1
  replicate=$2
  replicate_dir=$(get_replicate_dir "${mutation_ratio}" "${replicate}")
	echo "${replicate_dir}/databases/themisto"
}

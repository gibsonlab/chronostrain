# ================= Themisto + mSWEEP settings
export THEMISTO_BIN_DIR=/home/youn/work/themisto/build/bin
export POPPUNK_REFSEQ_DIR=${DATA_DIR}/poppunk


get_themisto_db_dir()
{
  replicate=$1
  replicate_dir=$(get_replicate_dir "${replicate}")
	echo "${replicate_dir}/databases/themisto"
}

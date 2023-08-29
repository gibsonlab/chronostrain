# ================= Themisto + mSWEEP settings
export STRAINGST_REF_FILES=/mnt/e/semisynthetic_data/straingst_files


get_straingst_db_dir()
{
  replicate=$1
  replicate_dir=$(get_replicate_dir "${replicate}")
	echo "${replicate_dir}/databases/straingst"
}

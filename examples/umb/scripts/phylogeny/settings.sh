# Modify these before running.
export PROJECT_DIR="/home/lactis/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export NCBI_REFSEQ_DIR="/mnt/e/ref_genomes"

export PHYLOGENY_OUTPUT_DIR="/mnt/e/chronostrain/phylogeny"
export LOGDIR="/mnt/e/chronostrain/phylogeny/logs"
export CHRONOSTRAIN_DB_DIR="/mnt/e/chronostrain/umb_database"
export MULTI_ALIGN_DIR="${CHRONOSTRAIN_DB_DIR}/strain_alignments/"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/phylogeny/settings.sh"

# ========
export METAPHLAN_PKL_PATH="/home/lactis/anaconda3/envs/metaphlan/lib/python3.7/site-packages/metaphlan/metaphlan_databases/mpa_vJan21_CHOCOPhlAnSGB_202103.pkl"

# ======== Chronostrain settings
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export CHRONOSTRAIN_DB_JSON_ALL="${CHRONOSTRAIN_DB_DIR}/database_all.json"
export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_JSON_ALL}"
export CHRONOSTRAIN_CACHE_DIR="${PHYLOGENY_OUTPUT_DIR}/cache"
export MULTIFASTA_FILE="all_markers.fasta"


# ========= Done.
echo "======================================================="
echo "Loaded shell settings from ${SETTINGS_PATH}."
echo "Chronostrain config: ${CHRONOSTRAIN_INI}"
echo "Logging config: ${CHRONOSTRAIN_LOG_INI}"
echo "Logging dir: ${LOGDIR}"
echo "======================================================="

check_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}
export check_program
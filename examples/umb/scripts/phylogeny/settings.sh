# Modify these before running.
export PROJECT_DIR="/home/youn/work/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export NCBI_REFSEQ_DIR="/mnt/e/ecoli_db/ref_genomes"

export PHYLOGENY_OUTPUT_DIR="/mnt/e/ecoli_db/phylogeny"
export LOGDIR="${PHYLOGENY_OUTPUT_DIR}/logs"

export CHRONOSTRAIN_DB_DIR="/mnt/e/ecoli_db/chronostrain_files"
#export MULTI_ALIGN_DIR="${CHRONOSTRAIN_DB_DIR}/strain_alignments"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/phylogeny/settings.sh"

# ======== MetaPhlAn settings
export METAPHLAN4_PKL_PATH="/mnt/e/metaphlan_databases/mpa_vJan21_CHOCOPhlAnSGB_202103/mpa_vJan21_CHOCOPhlAnSGB_202103.pkl"
export METAPHLAN3_PKL_PATH="/mnt/e/metaphlan_databases/mpa_v31_CHOCOPhlAn_201901/mpa_v31_CHOCOPhlAn_201901.pkl"

# ======== Chronostrain settings
export CHRONOSTRAIN_DB_JSON_ALL="${CHRONOSTRAIN_DB_DIR}/database_all.json"
export CHRONOSTRAIN_DB_JSON_PRUNED="${CHRONOSTRAIN_DB_DIR}/database_pruned.json"
export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_JSON_ALL}"

export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export CHRONOSTRAIN_CACHE_DIR="${PHYLOGENY_OUTPUT_DIR}/cache"


# ========= Done.
echo "======================================================="
echo "Loaded shell settings from ${SETTINGS_PATH}."
echo "Chronostrain config: ${CHRONOSTRAIN_INI}"
echo "Logging config: ${CHRONOSTRAIN_LOG_INI}"
echo "Logging dir: ${LOGDIR}"
echo "======================================================="

require_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}

require_file()
{
	path=$1
	if [ ! -f $path ]
	then
		echo "File ${path} not found."
		exit 1
	fi
}

export require_program
export require_file
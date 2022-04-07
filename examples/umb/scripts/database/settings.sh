# Modify these before running.
export PROJECT_DIR="/home/bromii/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/umb"
export DATA_DIR="/mnt/e/umb_data"
export LOGDIR="/mnt/d/ref_genomes/logs"
export CHRONOSTRAIN_DB_DIR="/mnt/d/chronostrain/umb_database"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/settings.sh"

# ========= NCBI RefSeqs
export NCBI_REFSEQ_DIR="/mnt/d/ref_genomes"
export REFSEQ_ALIGN_PATH="${CHRONOSTRAIN_DB_DIR}/strain_alignments/concatenation.fasta"
export BLAST_DB_DIR="${CHRONOSTRAIN_DB_DIR}/blast_db"
export BLAST_DB_NAME="esch_chrom"

# ======== Chronostrain settings
export CHRONOSTRAIN_DB_JSON_ALL="${CHRONOSTRAIN_DB_DIR}/database_all.json"
export CHRONOSTRAIN_DB_JSON_PRUNED="${CHRONOSTRAIN_DB_DIR}/database_pruned.json"
export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_JSON_PRUNED}"
export CHRONOSTRAIN_CACHE_DIR="${CHRONOSTRAIN_DB_DIR}/cache"
export MULTIFASTA_FILE="strain_markers.fasta"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"

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
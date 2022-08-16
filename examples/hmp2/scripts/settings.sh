# Modify these before running.
export PROJECT_DIR="/mnt/f/microbiome_tracking"
export CHRONOSTRAIN_DATA_DIR="/mnt/f/microbiome_tracking/data/hmp2"
export BASE_DIR="${PROJECT_DIR}/examples/hmp2"

# ======== Location of this file.
export SETTINGS_PATH="${BASE_DIR}/scripts/settings.sh"

# ========= NCBI RefSeqs
export NCBI_REFSEQ_DIR="${CHRONOSTRAIN_DATA_DIR}/ref_genomes"
export REFSEQ_ALIGN_PATH="${CHRONOSTRAIN_DATA_DIR}/strain_alignments/concatenation.fasta"

# ========= (Metaphlan-specific configuration.)
export METAPHLAN_PKL_PATH=/mnt/e/metaphlan/mpa_v30_CHOCOPhlAn_201901.pkl

# Location for reads and outputs.
export OUTPUT_DIR="${CHRONOSTRAIN_DATA_DIR}/output"
export PHYLOGENY_OUTPUT_DIR="${OUTPUT_DIR}/phylogeny"
export READS_DIR="${CHRONOSTRAIN_DATA_DIR}/reads"
export CHRONOSTRAIN_OUTPUT_DIR="${OUTPUT_DIR}/chronostrain"

# The chronostrain input index.
export INPUT_INDEX_FILENAME="inputs.csv"

# sratools prefetch/sample output directories.
export SRA_PREFETCH_DIR="${CHRONOSTRAIN_DATA_DIR}/prefetch"
export FASTERQ_TMP_DIR="${CHRONOSTRAIN_DATA_DIR}/fasterq-tmp"
export SAMPLES_DIR="${CHRONOSTRAIN_DATA_DIR}/samples"

# =========
export HMP2_CSV_PATH="${SAMPLES_DIR}/hmp2_metadata.csv"

# ========= Kneaddata
export KNEADDATA_DB_DIR="/mnt/e/kneaddata_db"
export NEXTERA_ADAPTER_PATH="/PHShome/yk847/.conda/envs/chronostrain/share/trimmomatic/adapters/NexteraPE-PE.fa"
export TRIMMOMATIC_DIR="/PHShome/yk847/.conda/envs/chronostrain/share/trimmomatic-0.39-2"

# ========= Chronostrain settings
export CHRONOSTRAIN_NUM_ITERS=50
export CHRONOSTRAIN_NUM_SAMPLES=150
export CHRONOSTRAIN_BATCH_SZ=2000
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.001
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"

export CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DATA_DIR}/database_all.json"
#export CHRONOSTRAIN_DB_JSON_PRUNED="${CHRONOSTRAIN_DATA_DIR}/database_pruned.json"
export MULTIFASTA_FILE="strain_markers.fasta"
export CHRONOSTRAIN_DB_DIR="${CHRONOSTRAIN_DATA_DIR}/database"

# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export LOGDIR="${CHRONOSTRAIN_DATA_DIR}/logs"

# ========= Done.
if [[ "${1}" == "verbose" ]]; then
	echo "======================================================="
	echo "Loaded shell settings from ${SETTINGS_PATH}."
	echo "Chronostrain config: ${CHRONOSTRAIN_INI}"
	echo "Logging config: ${CHRONOSTRAIN_LOG_INI}"
	echo "Logging dir: ${LOGDIR}"
	echo "======================================================="
fi

check_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 255;
	}
}
export check_program
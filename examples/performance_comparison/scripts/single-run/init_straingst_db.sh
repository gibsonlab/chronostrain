set -e
source settings.sh

mkdir -p ${STRAINGE_DB_DIR}

# Clear contents of the directory.
db_file_fmt=${STRAINGE_DB_DIR}/*
for f in $db_file_fmt;
do
	if [[ "$f" != "$db_file_fmt" ]]
	then
		echo "Deleting existing file: ${f}"
		rm ${f}
	fi
done

echo "**** Generating marker files. ****"
echo "Target dir: ${STRAINGE_DB_DIR}"
export CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/db_init/straingst.log"
python ${BASE_DIR}/scripts/helpers/chronostrain_markers_to_fasta.py \
-o ${STRAINGE_DB_DIR} \
-e ".fasta"

KMERIZED_FILES=""
echo "**** Kmerizing marker files. ****"
for genome_file in ${STRAINGE_DB_DIR}/*.fasta
do
	bn=$(basename ${genome_file%.fasta})
	echo "straingst kmerize [basename=${bn}]"

	kmerized_genome_file="${STRAINGE_DB_DIR}/${bn}.hdf5"
	straingst kmerize \
		-o ${kmerized_genome_file} \
		$genome_file

	KMERIZED_FILES="${KMERIZED_FILES}${kmerized_genome_file} "
done

straingst createdb \
-o ${STRAINGE_DB_PATH} \
$KMERIZED_FILES



set -e
source settings.sh

mkdir -p ${STRAINGE_GENOMES_DB_DIR}

# Clear contents of the directory.
db_file_fmt=${STRAINGE_GENOMES_DB_DIR}/*
for f in $db_file_fmt;
do
	if [[ "$f" != "$db_file_fmt" ]]
	then
		echo "Deleting existing file: ${f}"
		rm ${f}
	fi
done

echo "**** Generating marker files. ****"
echo "Target dir: ${STRAINGE_GENOMES_DB_DIR}"
cp ${CHRONOSTRAIN_DB_DIR}/AE014075.1.fasta ${STRAINGE_GENOMES_DB_DIR}/AE014075.1.fasta
cp ${CHRONOSTRAIN_DB_DIR}/CP000243.1.fasta ${STRAINGE_GENOMES_DB_DIR}/CP000243.1.fasta

KMERIZED_FILES=""
echo "**** Kmerizing marker files. ****"
for genome_file in ${STRAINGE_GENOMES_DB_DIR}/*.fasta
do
	bn=$(basename ${genome_file%.fasta})
	echo "straingst kmerize [basename=${bn}]"

	kmerized_genome_file="${STRAINGE_GENOMES_DB_DIR}/${bn}.hdf5"
	straingst kmerize \
		-o ${kmerized_genome_file} \
		$genome_file

	KMERIZED_FILES="${KMERIZED_FILES}${kmerized_genome_file} "  # Note: the trailing space is important here.
done

straingst createdb -o ${STRAINGE_GENOMES_DB_PATH} ${KMERIZED_FILES}



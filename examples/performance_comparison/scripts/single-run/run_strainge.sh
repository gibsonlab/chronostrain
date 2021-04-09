source settings.sh

mkdir -p ${STRAINGE_OUTPUT_DIR}

INPUT_FILE_ARGS=""
INPUT_TIME_ARGS=""

for t in "1.0" "2.0" "2.5" "5.0"
do
	READ_INPUT_FILE="${READS_DIR}/filtered/reads_${t}.fq"
	READ_KMERIZED_FILE="${STRAINGE_OUTPUT_DIR}/reads_${t}.hdf5"
	OUTPUT_FILE="${STRAINGE_OUTPUT_DIR}/reads_${t}.tsv"

	straingst kmerize \
		-k 23 \
		-o $READ_KMERIZED_FILE \
		${READ_INPUT_FILE}

	straingst run \
		-o "${OUTPUT_FILE}" \
		${STRAINGE_DB_PATH} \
		${READ_KMERIZED_FILE}

	INPUT_FILE_ARGS="${INPUT_FILE_ARGS} -i ${OUTPUT_FILE}"
	INPUT_TIME_ARGS="${INPUT_TIME_ARGS} -t ${t}"
done

python ${BASE_DIR}/scripts/helpers/strainge_to_ra.py \
${INPUT_FILE_ARGS} \
${INPUT_TIME_ARGS} \
-o ${STRAINGE_OUTPUT_DIR}/abundances.csv \
--strain_trim .fa.gz

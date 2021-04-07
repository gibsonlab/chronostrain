source settings.sh

CONSTRAIN_EXEC_DIR="/home/younhun/constrains"
CONSTRAIN_OUTPUT_DIR="${TRIAL_DIR}/output/constrains"
CONSTRAIN_CFG_PATH="${CONSTRAIN_OUTPUT_DIR}/config.conf"
METAPHLAN2_SCRIPT_PATH="/home/younhun/metaphlan2/metaphlan2.py"
NUM_CORES=1

mkdir -p $CONSTRAIN_OUTPUT_DIR

# =============== Generate the ConStrain configuration file.
> $CONSTRAIN_CFG_PATH
for f in $READS_DIR/*.fastq
do
	bn=$(basename ${f%.fastq})
	echo "//" >> $CONSTRAIN_CFG_PATH
	echo "sample: ${bn}" >> $CONSTRAIN_CFG_PATH
	echo "fq: ${f}" >> $CONSTRAIN_CFG_PATH
done

python2 $CONSTRAIN_EXEC_DIR/ConStrains.py \
-c $CONSTRAIN_CFG_PATH \
-o $CONSTRAIN_OUTPUT_DIR \
--metaphlan2 $METAPHLAN2_SCRIPT_PATH \
-t $NUM_CORES

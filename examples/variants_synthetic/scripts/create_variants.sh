set -e
source settings.sh
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain_init.ini"

echo "Initializing chronostrain files."
echo "Using INI config override: ${CHRONOSTRAIN_INI}"
python $BASE_DIR/helpers/initialize_chronostrain.py

echo "Creating variants from ${variants_json}."
variants_json="${BASE_DIR}/files/variants.json"
python ${BASE_DIR}/helpers/create_variants.py \
-i ${variants_json} \
-o ${CHRONOSTRAIN_DB_DIR}
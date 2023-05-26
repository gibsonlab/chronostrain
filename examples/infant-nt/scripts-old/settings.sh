# Modify these before running.
export PROJECT_DIR="/home/youn/chronostrain"
export BASE_DIR="${PROJECT_DIR}/examples/caesarian"
export DATA_DIR="/mnt/e/caesarian_data"
export OUTPUT_DIR="/mnt/e/caesarian_data/outputs"


# ========= Chronostrain settings
export INFERENCE_SEED=31415
export CHRONOSTRAIN_NUM_ITERS=100
export CHRONOSTRAIN_NUM_SAMPLES=100
export CHRONOSTRAIN_READ_BATCH_SZ=10000
export CHRONOSTRAIN_NUM_EPOCHS=1000
export CHRONOSTRAIN_DECAY_LR=0.25
export CHRONOSTRAIN_LR=0.0005
export CHRONOSTRAIN_LOSS_TOL=1e-5
export CHRONOSTRAIN_LR_PATIENCE=5
export CHRONOSTRAIN_MIN_LR=1e-6
export CHRONOSTRAIN_OUTPUT_FILENAME="abundances.out"


# ========= (Example-specific configuration. No need to modify below this line, unless you really want it changed.)
export CHRONOSTRAIN_DB_JSON="${BASE_DIR}/files/database.json"
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
export CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
export CHRONOSTRAIN_DB_DIR="/home/youn/.chronostrain/databases"
export CHRONOSTRAIN_CACHE_DIR="/home/youn/.chronostrain/cache"


require_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}

export require_program

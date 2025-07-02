LLM_MODEL=$1  
AGENT="CodeActAgent"
EVAL_LIMIT=300
MAX_INTERACTIONS=50
NUM_WORKERS=1
DATASET="princeton-nlp/SWE-bench_Lite"


SAFE_DATASET="${DATASET//\//_}"
LOG_PATH="logs/run_infer/${SAFE_DATASET}/${LLM_MODEL}/${AGENT}"
mkdir -p "$LOG_PATH"

LOG_FILE="${LOG_PATH}/log_$(date +%Y%m%d_%H%M%S).log"

DEBUG=1 ./evaluation/benchmarks/swe_bench/scripts/$2 \
    "$LLM_MODEL" \
    HEAD \
    "$AGENT" \
    "$EVAL_LIMIT" \
    "$MAX_INTERACTIONS" \
    "$NUM_WORKERS" \
    "$DATASET" \
    test 2>&1 | tee "$LOG_FILE"

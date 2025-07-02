# LLM_MODEL="llm.qwen3_32b" #"llm.gpt_4_1"
LLM_MODEL="llm.claude-3_7_sonnet_20250219" #LLM_MODEL="llm.qwen3_32b" #"llm.gpt_4_1"
AGENT="CodeActAgent"
EVAL_LIMIT=300
MAX_INTERACTIONS=100
NUM_WORKERS=1
DATASET="princeton-nlp/SWE-bench_Lite"


SAFE_DATASET="${DATASET//\//_}"
LOG_PATH="logs/run_infer/${SAFE_DATASET}/${LLM_MODEL}/${AGENT}"
mkdir -p "$LOG_PATH"

LOG_FILE="${LOG_PATH}/log_$(date +%Y%m%d_%H%M%S).log"

DEBUG=1 ./evaluation/benchmarks/swe_bench/scripts/run_infer_plain.sh \
    "$LLM_MODEL" \
    HEAD \
    "$AGENT" \
    "$EVAL_LIMIT" \
    "$MAX_INTERACTIONS" \
    "$NUM_WORKERS" \
    "$DATASET" \
    test 2>&1 | tee "$LOG_FILE"

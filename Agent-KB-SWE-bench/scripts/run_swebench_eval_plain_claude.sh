#DATA_SOURCE="./evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/gpt-4.1_maxiter_50_N_v0.31.0-plain-run_1/output.jsonl"
DATA_SOURCE='./evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-plain-run_1/output.jsonl'
# DATA_SOURCE='./evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/gpt-4o_maxiter_50_N_v0.31.0-plain-run_1/output.jsonl'


DATA_SOURCE_DIR=$(dirname "${DATA_SOURCE}")  
MODEL_CONFIG=$(basename "${DATA_SOURCE_DIR}")  
AGENT_DIR=$(dirname "${DATA_SOURCE_DIR}")  
AGENT=$(basename "${AGENT_DIR}")  
DATASET_DIR=$(dirname "${AGENT_DIR}") 
DATASET=$(basename "${DATASET_DIR}")  


LOG_PATH="logs/run_evaluation/${DATASET}/${AGENT}/${MODEL_CONFIG}"
mkdir -p "${LOG_PATH}"


LOG_FILE="${LOG_PATH}/log_$(date +%Y%m%d_%H%M%S).log"


./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh "${DATA_SOURCE}" >> "${LOG_FILE}"

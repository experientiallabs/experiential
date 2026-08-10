#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
STEP="${STEP:-100}"
case "$STEP" in
  25|50|100|200) ;;
  *) echo "STEP must be 25, 50, 100, or 200" >&2; exit 2 ;;
esac

AXO_ROOT="$XROOT/axolotl-sft"
SCRIPTS="$AXO_ROOT/scripts"
EVAL_CODE="$XROOT/eval-code-b0d0568-run6"
HERE="$EVAL_CODE/xtoken-ops/tblite-eval-9b"
HARBOR_VENV=/scratch/rebench10/uvtools/datacurve-pier
VALIDATION="$XROOT/next-sft-candidates-v2/checkpoint-validation-merged-step${STEP}.json"
SERVER_SESSION="qwen35-4b-merged-step${STEP}-seeds-serve"
PORT=8122
BASE_MODEL=qwen35-4b-base
RUN_PREFIX=qwen35-4b-glm52-merged-realverified-sft-lr1e5-r64-seed
SEEDS=(20260809 20260810)

test "$(git -C "$EVAL_CODE" rev-parse HEAD)" = b0d05686f19646771d00ce5d76d1b42edfb8aced
test -z "$(git -C "$EVAL_CODE" status --porcelain)"
test -x "$HARBOR_VENV/bin/python"

"$AXO_ROOT/venv/bin/python" "$SCRIPTS/validate_candidate_sft_checkpoints.py" \
  --root "$AXO_ROOT/checkpoints" --run-prefix "$RUN_PREFIX" \
  --seeds "${SEEDS[@]}" --steps "$STEP" --out "$VALIDATION"

adapter_specs=""
for seed in "${SEEDS[@]}"; do
  name="qwen35-4b-glm52-merged-seed${seed}-step${STEP}"
  directory="$AXO_ROOT/checkpoints/${RUN_PREFIX}${seed}/checkpoint-${STEP}"
  adapter_specs+="${adapter_specs:+ }${name}=${directory}"
done

export XROOT
export ADAPTER_SPECS="$adapter_specs"
export SESSION="$SERVER_SESSION"
export PORT
export HOST=0.0.0.0
export CUDA_DEVICES=0,1
export DP_SIZE=2
# mini-swe-agent keeps its logical benchmark envelope at 65,536 tokens, while
# the Qwen chat template can add two renderer tokens at the boundary.
export MAXLEN=65538
export MAX_NUM_SEQS=32
export GPU_UTIL=0.85
export LOG_DIR="$AXO_ROOT/eval-logs/merged-step${STEP}-seeds"
export RUNTIME_DIR="$AXO_ROOT/eval-runtime/merged-step${STEP}-seeds"
bash "$SCRIPTS/serve_named_lora_set_4b.sh" start

run_pair() {
  local seed="$1"
  local arm="merged-seed${seed}-step${STEP}"
  local served="qwen35-4b-glm52-merged-seed${seed}-step${STEP}"
  local root="$XROOT/merged-step${STEP}-seed${seed}-tblite-canary10-seed0-run1"
  local prefix="qwen35-4b-merged-seed${seed}-step${STEP}-canary10-seed0"
  local log="$XROOT/logs/${prefix}.log"

  test ! -e "$root/jobs"
  mkdir -p "$root/runtime" "$XROOT/logs"
  export HERE
  export EVAL_ROOT="$root"
  export STORAGE_ROOT="$XROOT"
  export HARBOR_VENV
  export DOCKER_CONFIG=/home/azureuser/.docker
  export BASE_SNAP=/scratch/xtoken-offline-9b-20260727/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  export HOST_ADDR=172.16.0.4
  export API_BASE_LOCAL="http://127.0.0.1:${PORT}/v1"
  export API_BASE_AGENT="http://172.16.0.4:${PORT}/v1"
  export CUDA_DEVICES=0,1
  export TP=1
  export DP_SIZE=2
  # Endpoint assertion reflects the serving-only renderer guard above.
  export MAXLEN=65538
  export N_RUNS=1
  export N_TASKS=10
  export TASK_NAMES=
  export CONCURRENCY=2
  export SEED_BASE=0
  export EXPECTED_DOCKER_ROOT_PREFIX=/scratch/
  export BASE_MODEL
  export JOB_PREFIX="$prefix"
  export ADAPTER_ARM="$arm"
  export ADAPTER_MODEL="$served"
  sg docker -c "bash '$SCRIPTS/run_tblite_canary_pair.sh' 2>&1 | tee '$log'"

  "$HARBOR_VENV/bin/python" "$SCRIPTS/compare_tblite_paired.py" \
    --base "$root/jobs/${prefix}-base-run1" \
    --adapter "$root/jobs/${prefix}-${arm}-run1" \
    --out "$root/paired-vs-base-canary10.json"
  "$HARBOR_VENV/bin/python" -c \
    'import json,sys; row=json.load(open(sys.argv[1])); assert row["task_count"] == 10' \
    "$root/paired-vs-base-canary10.json"
}

run_pair 20260809
run_pair 20260810

"$HARBOR_VENV/bin/python" "$SCRIPTS/select_candidate_canary.py" \
  --step "$STEP" \
  --input "20260809=$XROOT/merged-step${STEP}-seed20260809-tblite-canary10-seed0-run1/paired-vs-base-canary10.json" \
  --input "20260810=$XROOT/merged-step${STEP}-seed20260810-tblite-canary10-seed0-run1/paired-vs-base-canary10.json" \
  --out "$XROOT/merged-step${STEP}-two-seed-canary-gate.json"

touch "$XROOT/merged-step${STEP}-two-seed-canaries.complete"

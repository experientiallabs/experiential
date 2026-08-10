#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
STEP="${STEP:?STEP must be 25, 50, 100, or 200}"
case "$STEP" in
  25|50|100|200) ;;
  *) echo "STEP must be 25, 50, 100, or 200" >&2; exit 2 ;;
esac

TRAIN_SEED=20260809
AXO_ROOT="$XROOT/axolotl-sft"
SCRIPTS="$AXO_ROOT/scripts"
EVAL_CODE="$XROOT/eval-code-b0d0568-run6"
HERE="$EVAL_CODE/xtoken-ops/tblite-eval-9b"
HARBOR_VENV=/scratch/rebench10/uvtools/datacurve-pier
HARBOR_PY="$HARBOR_VENV/bin/python"
SERVER_SESSION="qwen35-4b-candidate-step${STEP}-seeds-serve"
GATE="$XROOT/candidate-step${STEP}-two-seed-canary-gate.json"
ADAPTER_ARM="candidate-seed${TRAIN_SEED}-step${STEP}"
ADAPTER_MODEL="qwen35-4b-glm52-candidate-seed${TRAIN_SEED}-step${STEP}"
BASE_MODEL=qwen35-4b-base
PORT=8122
REPORTS=()

test "$(git -C "$EVAL_CODE" rev-parse HEAD)" = b0d05686f19646771d00ce5d76d1b42edfb8aced
test -z "$(git -C "$EVAL_CODE" status --porcelain)"
test -x "$HARBOR_PY"
test -s "$GATE"
"$HARBOR_PY" - "$GATE" "$STEP" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1]))
assert row["checkpoint_step"] == int(sys.argv[2])
assert row["credible_direction"] is True
assert row["predeclared_primary_if_credible"] == f"seed20260809-step{sys.argv[2]}"
PY
tmux has-session -t "$SERVER_SESSION"
curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models" | \
  "$HARBOR_PY" -c 'import json,sys; names={x["id"] for x in json.load(sys.stdin)["data"]}; assert sys.argv[1] in names' \
  "$ADAPTER_MODEL"

for eval_seed in 0 1 2; do
  root="$XROOT/candidate-step${STEP}-seed${TRAIN_SEED}-tblite-full100-eval-seed${eval_seed}-run1"
  prefix="qwen35-4b-candidate-seed${TRAIN_SEED}-step${STEP}-full100-eval-seed${eval_seed}"
  report="$root/paired-vs-base-full100.json"
  REPORTS+=("$report")

  if test -s "$report"; then
    "$HARBOR_PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["task_count"] == 100' "$report"
    continue
  fi

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
  # The logical agent envelope remains 65,536; Qwen rendering needs two
  # serving-only guard tokens at the boundary.
  export MAXLEN=65538
  export N_RUNS=1
  export N_TASKS=100
  export TASK_NAMES=
  export CONCURRENCY=2
  export SEED_BASE="$eval_seed"
  export EXPECTED_DOCKER_ROOT_PREFIX=/scratch/
  export BASE_MODEL
  export JOB_PREFIX="$prefix"
  export ADAPTER_ARM
  export ADAPTER_MODEL
  sg docker -c "bash '$SCRIPTS/run_tblite_canary_pair.sh' 2>&1 | tee '$XROOT/logs/${prefix}.log'"

  "$HARBOR_PY" "$SCRIPTS/compare_tblite_paired.py" \
    --base "$root/jobs/${prefix}-base-run1" \
    --adapter "$root/jobs/${prefix}-${ADAPTER_ARM}-run1" \
    --out "$report"
  "$HARBOR_PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["task_count"] == 100' "$report"
done

AGGREGATE="$XROOT/candidate-step${STEP}-seed${TRAIN_SEED}-tblite-repeated-eval-seeds0-2.json"
"$HARBOR_PY" "$SCRIPTS/aggregate_tblite_repeated.py" \
  --input "${REPORTS[0]}" --input "${REPORTS[1]}" --input "${REPORTS[2]}" \
  --expected-task-count 100 --out "$AGGREGATE"
touch "$XROOT/candidate-step${STEP}-seed${TRAIN_SEED}-tblite-repeated.complete"

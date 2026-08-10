#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
STEP="${STEP:-100}"
SEED="${SEED:?SEED is required}"
AXO_ROOT="$XROOT/axolotl-sft"
SCRIPTS="$AXO_ROOT/scripts"
EVAL_CODE="$XROOT/eval-code-b0d0568-run6"
HERE="$EVAL_CODE/xtoken-ops/tblite-eval-9b"
HARBOR_VENV=/scratch/rebench10/uvtools/datacurve-pier
PORT=8122
BASE_MODEL=qwen35-4b-base

test "$(git -C "$EVAL_CODE" rev-parse HEAD)" = b0d05686f19646771d00ce5d76d1b42edfb8aced
test -z "$(git -C "$EVAL_CODE" status --porcelain)"
test -x "$HARBOR_VENV/bin/python"
tmux has-session -t "qwen35-4b-candidate-step${STEP}-seeds-serve"

arm="candidate-seed${SEED}-step${STEP}"
served="qwen35-4b-glm52-candidate-seed${SEED}-step${STEP}"
root="$XROOT/candidate-step${STEP}-seed${SEED}-tblite-canary10-seed0-run1"
prefix="qwen35-4b-candidate-seed${SEED}-step${STEP}-canary10-seed0"
log="$XROOT/logs/${prefix}.resume.log"

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

first="$XROOT/candidate-step${STEP}-seed20260809-tblite-canary10-seed0-run1/paired-vs-base-canary10.json"
second="$XROOT/candidate-step${STEP}-seed20260810-tblite-canary10-seed0-run1/paired-vs-base-canary10.json"
if test -s "$first" && test -s "$second"; then
  "$HARBOR_VENV/bin/python" "$SCRIPTS/select_candidate_canary.py" \
    --step "$STEP" \
    --input "20260809=$first" \
    --input "20260810=$second" \
    --out "$XROOT/candidate-step${STEP}-two-seed-canary-gate.json"
  touch "$XROOT/candidate-step${STEP}-two-seed-canaries.complete"
fi

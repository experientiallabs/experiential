#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
DATA_ROOT="$XROOT/next-sft-candidates-v1"
PYTHON="$XROOT/runtime/venv-harbor-tb2/bin/python"
E2B_ENV="$XROOT/runtime/credentials/e2b.env"
TMAX_SOURCE="$XROOT/assets/tmax-source-7387d2f91423"
CODE_ROOT="$XROOT/eval-code-b0d0568-run6"
TASKS_ROOT="$XROOT/assets/tmax15k-subset-1600/tasks"
AUDIT_DATASET="$DATA_ROOT/sft-data/candidate192.audit.jsonl"
REPLAY_SCRIPT="$AXO_ROOT/scripts/replay_admitted_teacher_trajectories.py"
OUT="$DATA_ROOT/teacher-replay/run1"
LOG="$DATA_ROOT/logs/teacher-replay-run1.log"

for required in "$PYTHON" "$E2B_ENV" "$AUDIT_DATASET" "$REPLAY_SCRIPT"; do
  test -r "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
test -d "$TMAX_SOURCE"
test -d "$TASKS_ROOT"
test "$(jq -r 'select(.admission.selected_for_sft != true) | .source_row_index' "$AUDIT_DATASET" | wc -l)" -eq 0

set -a
# shellcheck source=/dev/null
. "$E2B_ENV"
set +a
test -n "${E2B_API_KEY:-}"
export PYTHONPATH="$TMAX_SOURCE:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$(dirname "$LOG")"

"$PYTHON" "$REPLAY_SCRIPT" \
  --audit-dataset "$AUDIT_DATASET" \
  --tasks "$TASKS_ROOT" \
  --out "$OUT" \
  --template-alias tmax15k-full-da54e6370473 \
  --run-id qwen35-4b-glm52-candidate192-teacher-real-verifier-run1 \
  --concurrency 8 \
  --sandbox-timeout-s 3600 \
  --bash-timeout-s 120 \
  --resume \
  2>&1 | tee -a "$LOG"

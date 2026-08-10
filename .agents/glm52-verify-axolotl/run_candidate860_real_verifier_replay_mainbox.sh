#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
CODE_COMMIT="${CODE_COMMIT:?set CODE_COMMIT to the exact deployed 40-character Git SHA}"
AXO_ROOT="$XROOT/axolotl-sft"
DATA_ROOT="$XROOT/next-sft-candidates-v2"
PYTHON="$XROOT/runtime/venv-harbor-tb2/bin/python"
E2B_ENV="$XROOT/runtime/credentials/e2b.env"
TMAX_SOURCE="$XROOT/assets/tmax-source-7387d2f91423"
CODE_ROOT="$XROOT/eval-code-b0d0568-run6"
TASKS_ROOT="$XROOT/assets/tmax15k-subset-1600/tasks"
CORPUS=/scratch/xtoken-offline-9b-20260727/tmax/offline-c43abfc-20260802/corpus-40960.jsonl
SOURCE_SHA256=7220f5d58e41933e38c46a29eee37ff4da4a21e8901ea27f9ead624cc6df911a
CANDIDATES="$DATA_ROOT/candidate860-new.jsonl"
CANDIDATE_MANIFEST="$DATA_ROOT/candidate860-new.manifest.json"
PRIOR_CANDIDATES="$XROOT/next-sft-candidates-v1/candidate192.jsonl"
EXPANDED_CANDIDATES="$DATA_ROOT/candidate1052.jsonl"
EXPANDED_MANIFEST="$DATA_ROOT/candidate1052.manifest.json"
AUDIT="$DATA_ROOT/replay/candidate860.replay-audit-v2.jsonl"
AUDIT_MANIFEST="$DATA_ROOT/replay/candidate860.replay-audit-v2.manifest.json"
BUILD_SCRIPT="${BUILD_SCRIPT:-$AXO_ROOT/scripts/build_replay_candidate_audit.py}"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-$AXO_ROOT/scripts/replay_admitted_teacher_trajectories.py}"
OUT="$DATA_ROOT/teacher-replay/run1"
LOG="$DATA_ROOT/logs/teacher-replay-run1.log"

for required in "$PYTHON" "$E2B_ENV" "$CORPUS" "$CANDIDATES" \
  "$CANDIDATE_MANIFEST" "$PRIOR_CANDIDATES" "$EXPANDED_CANDIDATES" \
  "$EXPANDED_MANIFEST" "$BUILD_SCRIPT" "$REPLAY_SCRIPT"; do
  test -r "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
test -d "$TMAX_SOURCE"
test -d "$TASKS_ROOT"

if test ! -e "$AUDIT" && test ! -e "$AUDIT_MANIFEST"; then
  "$PYTHON" "$BUILD_SCRIPT" \
    --corpus "$CORPUS" \
    --source-sha256 "$SOURCE_SHA256" \
    --candidates "$CANDIDATES" \
    --candidate-manifest "$CANDIDATE_MANIFEST" \
    --prior-candidates "$PRIOR_CANDIDATES" \
    --expanded-candidates "$EXPANDED_CANDIDATES" \
    --expanded-manifest "$EXPANDED_MANIFEST" \
    --code-commit "$CODE_COMMIT" \
    --output "$AUDIT" \
    --manifest "$AUDIT_MANIFEST"
fi
test -r "$AUDIT"
test -r "$AUDIT_MANIFEST"
test "$(wc -l < "$AUDIT")" -eq 860
test "$(jq -r 'select(.admission.selected_for_replay != true or .admission.selected_for_sft != false) | .source_row_index' "$AUDIT" | wc -l)" -eq 0

set -a
# shellcheck source=/dev/null
. "$E2B_ENV"
set +a
test -n "${E2B_API_KEY:-}"
export PYTHONPATH="$TMAX_SOURCE:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$(dirname "$LOG")"

"$PYTHON" "$REPLAY_SCRIPT" \
  --audit-dataset "$AUDIT" \
  --tasks "$TASKS_ROOT" \
  --out "$OUT" \
  --template-alias tmax15k-full-da54e6370473 \
  --run-id qwen35-4b-glm52-candidate860-teacher-real-verifier-run1 \
  --concurrency 8 \
  --sandbox-timeout-s 3600 \
  --bash-timeout-s 120 \
  --code-commit "$CODE_COMMIT" \
  --resume \
  2>&1 | tee -a "$LOG"

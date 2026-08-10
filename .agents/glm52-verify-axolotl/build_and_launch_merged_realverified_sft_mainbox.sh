#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
DATA_V1="$XROOT/next-sft-candidates-v1/sft-data"
DATA_V2="$XROOT/next-sft-candidates-v2"
SFT_ROOT="$DATA_V2/sft-data"
CODE_COMMIT="${CODE_COMMIT:?set CODE_COMMIT to the exact deployed 40-character Git SHA}"
CODE_ROOT="$DATA_V2/code-${CODE_COMMIT:0:8}"
PYTHON="$XROOT/runtime/venv-harbor-tb2/bin/python"
AXO_PYTHON="$AXO_ROOT/venv/bin/python"
LOG="$DATA_V2/logs/candidate1052-build-launch.log"

PRIMARY_REPLAY="$DATA_V2/teacher-replay/run1/episodes"
PRIMARY_SUMMARY="$DATA_V2/teacher-replay/run1/summary.json"
AUDIT_860="$DATA_V2/replay/candidate860.replay-audit-v2.jsonl"
QWEN_860_PENDING="$SFT_ROOT/candidate860.pending.qwen-materialized.jsonl"
RECOVERY_ROOTS=(
  "$DATA_V2/teacher-replay/recovery-task_008280-recorded-timeout-run1/episodes"
  "$DATA_V2/teacher-replay/recovery-recorded-timeout4-run1/episodes"
  "$DATA_V2/teacher-replay/recovery-recorded-timeout2-late-run1/episodes"
  "$DATA_V2/teacher-replay/recovery-verifier-timeout2-verify1200-run2/episodes"
  "$DATA_V2/teacher-replay/recovery-verifier-timeout1-late-verify1200-run1/episodes"
  "$DATA_V2/teacher-replay/recovery-exit143-timeout1-bash120-run2/episodes"
)
RECOVERY_EXPECTED=(1 4 2 2 1 1)

FILTERED_AUDIT="$SFT_ROOT/candidate860.realverified-perfect.audit.jsonl"
FILTERED_QWEN="$SFT_ROOT/candidate860.realverified-perfect.qwen-materialized.jsonl"
FILTERED_LEDGER="$SFT_ROOT/candidate860.realverified-perfect.ledger.jsonl"
FILTERED_MANIFEST="$SFT_ROOT/candidate860.realverified-perfect.manifest.json"
MERGED_AUDIT="$SFT_ROOT/candidate1052.merged.realverified-perfect.audit.jsonl"
MERGED_QWEN="$SFT_ROOT/candidate1052.merged.realverified-perfect.qwen-materialized.jsonl"
MERGED_MANIFEST="$SFT_ROOT/candidate1052.merged.realverified-perfect.manifest.json"
MASK_AUDIT="$SFT_ROOT/candidate1052.merged.realverified-perfect.mask-audit.json"
TBLITE_DISJOINT="$SFT_ROOT/candidate1052.merged.realverified-perfect.disjoint-tblite.json"
TB2_DISJOINT="$SFT_ROOT/candidate1052.merged.realverified-perfect.disjoint-tb2.json"

FILTER_SCRIPT="$CODE_ROOT/build_real_verifier_filtered_sft.py"
MERGE_SCRIPT="$CODE_ROOT/merge_realverified_sft_bundles.py"
MASK_SCRIPT="$AXO_ROOT/scripts/audit_axolotl_masks.py"
DISJOINT_SCRIPT="$AXO_ROOT/scripts/verify_tblite_training_disjoint.py"
CONFIG="$AXO_ROOT/configs/qwen35_4b_merged_realverified_sft_mainbox.yaml"
TRAIN_SCRIPT="$AXO_ROOT/scripts/run_merged_realverified_sft_mainbox.sh"
MONITOR_SCRIPT="$AXO_ROOT/scripts/monitor_candidate192_training.py"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

wait_for_session_end() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    printf '%s waiting for session %s\n' "$(timestamp)" "$session"
    sleep 60
  done
}

for required in "$PYTHON" "$AXO_PYTHON" "$AUDIT_860" "$QWEN_860_PENDING" \
  "$PRIMARY_SUMMARY" "$FILTER_SCRIPT" "$MERGE_SCRIPT" "$MASK_SCRIPT" \
  "$DISJOINT_SCRIPT" "$CONFIG" "$TRAIN_SCRIPT" "$MONITOR_SCRIPT"; do
  test -r "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]]

wait_for_session_end candidate860-replay-recovery-recorded-timeout4
wait_for_session_end candidate860-replay-recovery-recorded-timeout2-late

for index in "${!RECOVERY_ROOTS[@]}"; do
  root="${RECOVERY_ROOTS[$index]}"
  expected="${RECOVERY_EXPECTED[$index]}"
  test "$(find "$root" -mindepth 2 -maxdepth 2 -name replay_result.json -type f | wc -l)" -eq "$expected"
  test "$(find "$root" -mindepth 2 -maxdepth 2 -name replay_result.json -type f -exec jq -r 'select(.finished_at == null) | .task_id' {} + | wc -l)" -eq 0
done

for output in "$FILTERED_AUDIT" "$FILTERED_QWEN" "$FILTERED_LEDGER" \
  "$FILTERED_MANIFEST" "$MERGED_AUDIT" "$MERGED_QWEN" "$MERGED_MANIFEST" \
  "$MASK_AUDIT" "$TBLITE_DISJOINT" "$TB2_DISJOINT"; do
  test ! -e "$output" || { echo "refusing to overwrite: $output" >&2; exit 1; }
done

mkdir -p "$SFT_ROOT" "$(dirname "$LOG")"
recovery_args=()
for root in "${RECOVERY_ROOTS[@]}"; do
  recovery_args+=(--recovery-root "$root")
done
"$PYTHON" "$FILTER_SCRIPT" \
  --audit-dataset "$AUDIT_860" \
  --qwen-dataset "$QWEN_860_PENDING" \
  --replay-root "$PRIMARY_REPLAY" \
  --replay-summary "$PRIMARY_SUMMARY" \
  "${recovery_args[@]}" \
  --minimum-reward 1.0 \
  --output-audit "$FILTERED_AUDIT" \
  --output-qwen "$FILTERED_QWEN" \
  --output-ledger "$FILTERED_LEDGER" \
  --output-manifest "$FILTERED_MANIFEST"

"$PYTHON" "$MERGE_SCRIPT" \
  --input-bundle prior149 \
    "$DATA_V1/candidate192.realverified-perfect.audit.jsonl" \
    "$DATA_V1/candidate192.realverified-perfect.qwen-materialized.jsonl" \
    "$DATA_V1/candidate192.realverified-perfect.manifest.json" \
  --input-bundle candidate860 \
    "$FILTERED_AUDIT" "$FILTERED_QWEN" "$FILTERED_MANIFEST" \
  --output-audit "$MERGED_AUDIT" \
  --output-qwen "$MERGED_QWEN" \
  --output-manifest "$MERGED_MANIFEST"

ROWS="$(jq -r .rows "$MERGED_MANIFEST")"
test "$ROWS" -ge 850
test "$ROWS" -le 1052
export HF_HOME=/scratch/xtoken-offline-9b-20260727/cache/huggingface
"$AXO_PYTHON" "$MASK_SCRIPT" \
  --dataset "$MERGED_QWEN" \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --sequence-len 32768 \
  --output "$MASK_AUDIT"

"$PYTHON" "$DISJOINT_SCRIPT" \
  --training-audit "$MERGED_AUDIT" \
  --tblite-root /scratch/xtoken-offline-ce32k-20260728/tblite \
  --dataset-git-commit 5c37b41f00ce04719a4453061076ae9f46b74b7d \
  --expected-training-tasks "$ROWS" \
  --expected-tblite-tasks 100 \
  --out "$TBLITE_DISJOINT"

"$PYTHON" "$DISJOINT_SCRIPT" \
  --training-audit "$MERGED_AUDIT" \
  --tblite-root /scratch/xtoken-offline-9b-20260727/tb2/dataset/terminal-bench \
  --dataset-git-commit instruction-bundle-sha256:c49b8115ad86f7b3284b2b52558373d348c3478987a2c1e45393141f944871c7 \
  --expected-training-tasks "$ROWS" \
  --expected-tblite-tasks 89 \
  --out "$TB2_DISJOINT"

sha256sum "$FILTERED_AUDIT" "$FILTERED_QWEN" "$FILTERED_MANIFEST" \
  "$MERGED_AUDIT" "$MERGED_QWEN" "$MERGED_MANIFEST" "$MASK_AUDIT" \
  "$TBLITE_DISJOINT" "$TB2_DISJOINT"
touch "$DATA_V2/candidate1052-merged-realverified-build.complete"

while test ! -e "$XROOT/candidate-step25-two-seed-canaries.complete"; do
  printf '%s waiting for prior canary completion\n' "$(timestamp)"
  sleep 60
done
wait_for_session_end candidate-step25-two-seed-canaries

SERVER_SESSION=qwen35-4b-candidate-step25-seeds-serve
if tmux has-session -t "$SERVER_SESSION" 2>/dev/null; then
  printf '%s stopping completed canary server %s\n' "$(timestamp)" "$SERVER_SESSION"
  tmux send-keys -t "$SERVER_SESSION" C-c
  for _ in $(seq 1 12); do
    tmux has-session -t "$SERVER_SESSION" 2>/dev/null || break
    sleep 5
  done
  tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
fi

for _ in $(seq 1 30); do
  gpu0="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  gpu1="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  if test "$gpu0" -le 8192 && test "$gpu1" -le 8192; then
    break
  fi
  printf '%s waiting for GPU release gpu0=%s MiB gpu1=%s MiB\n' \
    "$(timestamp)" "$gpu0" "$gpu1"
  sleep 10
done
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')" -le 8192
test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')" -le 8192

for pair in "0 20260809" "1 20260810"; do
  read -r gpu seed <<<"$pair"
  session="glm52-merged-realverified-sft-seed${seed}"
  tmux has-session -t "$session" 2>/dev/null && { echo "session exists: $session" >&2; exit 1; }
  tmux new-session -d -s "$session" \
    "bash '$TRAIN_SCRIPT' '$gpu' '$seed' 2>&1 | tee '$AXO_ROOT/logs/${session}.launch.log'"
done

sleep 35
tmux has-session -t glm52-merged-realverified-sft-seed20260809
tmux has-session -t glm52-merged-realverified-sft-seed20260810

tmp="$(mktemp)"
crontab -l 2>/dev/null | \
  grep -Fv 'codex-candidate192-training-monitor' | \
  grep -Fv 'codex-merged-realverified-training-monitor' | \
  grep -Fv 'codex-candidate1052-build-launch-monitor' | \
  grep -Fv 'codex-candidate-step100-canary-monitor' | \
  grep -Fv 'codex-candidate-checkpoint-eval-advance' | \
  grep -Fv 'codex-candidate860-recovery-recorded-timeout4-monitor' | \
  grep -Fv 'codex-candidate860-recovery-recorded-timeout2-late-monitor' | \
  grep -Fv 'codex-candidate860-recovery-verifier-timeout1-late-monitor' \
  >"$tmp" || true
printf '*/2 * * * * %s %s --root %s --health-log %s --run-prefix %s --session-prefix %s >/dev/null 2>&1 # codex-merged-realverified-training-monitor\n' \
  "$AXO_PYTHON" "$MONITOR_SCRIPT" "$AXO_ROOT" \
  "$DATA_V2/monitor/merged-realverified-training-health.jsonl" \
  qwen35-4b-glm52-merged-realverified-sft-lr1e5-r64-seed \
  glm52-merged-realverified-sft-seed >>"$tmp"
crontab "$tmp"
rm -f "$tmp"
touch "$DATA_V2/candidate1052-merged-realverified-training.launched"
printf '%s launched two merged real-verifier SFT seeds\n' "$(timestamp)"

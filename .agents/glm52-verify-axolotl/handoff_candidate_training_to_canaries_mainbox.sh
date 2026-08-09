#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
SCRIPTS="$AXO_ROOT/scripts"
SESSION=candidate-step100-two-seed-canaries
SERVER_SESSION=qwen35-4b-candidate-step100-seeds-serve
LAUNCHER="$SCRIPTS/run_candidate_step100_seed_canaries_mainbox.sh"
MONITOR="$SCRIPTS/monitor_candidate_step100_canaries.py"
HEALTH_LOG="$XROOT/next-sft-candidates-v1/monitor/candidate-step100-canaries-health.jsonl"
MARKER="$XROOT/candidate-step100-two-seed-canaries.launched"
FAILED="$XROOT/candidate-step100-two-seed-canaries.preflight-failed"
LOG="$XROOT/logs/candidate-step100-two-seed-canaries.launch.log"
LOCK="$XROOT/runtime/candidate-step100-two-seed-canaries-handoff.lock"

mkdir -p "$XROOT/runtime" "$XROOT/logs" "$(dirname "$HEALTH_LOG")"
exec 9>"$LOCK"
flock -n 9 || exit 0
test ! -e "$MARKER" || exit 0
test ! -e "$FAILED" || exit 0

for seed in 20260809 20260810; do
  tmux has-session -t "glm52-candidate192-sft-seed${seed}" 2>/dev/null && exit 0
  run="qwen35-4b-glm52-candidate-realverified-sft-lr1e5-r64-seed${seed}"
  train_log="$AXO_ROOT/logs/${run}.log"
  checkpoint="$AXO_ROOT/checkpoints/${run}/checkpoint-200/adapter_model.safetensors"
  if ! grep -qi 'training completed!' "$train_log" || \
     ! grep -qi 'model successfully saved' "$train_log" || \
     ! test -s "$checkpoint"; then
    printf '%s incomplete seed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" >>"$LOG"
    exit 0
  fi
  if grep -Eqi 'traceback|out of memory|cuda oom|(^|[^a-z])nan([^a-z]|$)' "$train_log"; then
    printf '%s fatal training signal seed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" >>"$LOG"
    touch "$FAILED"
    exit 1
  fi
done

for gpu in 0 1; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  if test "$used" -gt 8192; then
    printf '%s waiting gpu=%s memory_mib=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu" "$used" >>"$LOG"
    exit 0
  fi
done

test -x "$LAUNCHER"
test -x "$MONITOR"
tmux has-session -t "$SERVER_SESSION" 2>/dev/null && {
  printf '%s unexpected server session already exists\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG"
  touch "$FAILED"
  exit 1
}
tmux new-session -d -s "$SESSION" "STEP=100 bash '$LAUNCHER' 2>&1 | tee '$LOG'"

tmp="$(mktemp)"
crontab -l 2>/dev/null | grep -Fv 'codex-candidate192-training-monitor' | \
  grep -Fv 'codex-candidate-step100-canary-monitor' >"$tmp" || true
printf '*/2 * * * * %s --root %s --health-log %s --step 100 # codex-candidate-step100-canary-monitor\n' \
  "$MONITOR" "$XROOT" "$HEALTH_LOG" >>"$tmp"
crontab "$tmp"
rm -f "$tmp"
touch "$MARKER"

sleep 35
tmux has-session -t "$SESSION"
printf '%s launched two-seed step100 held-out canaries\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG"

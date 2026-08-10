#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
SCRIPTS="$AXO_ROOT/scripts"
HARBOR_PY=/scratch/rebench10/uvtools/datacurve-pier/bin/python
LOCK="$XROOT/runtime/candidate-checkpoint-eval-advance.lock"
LOG="$XROOT/logs/candidate-checkpoint-eval-advance.log"

mkdir -p "$XROOT/runtime" "$XROOT/logs"
exec 9>"$LOCK"
flock -n 9 || exit 0

credible() {
  "$HARBOR_PY" - "$1" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1]))
print("1" if row["credible_direction"] else "0")
PY
}

install_canary_monitor() {
  local step="$1"
  local tmp
  tmp="$(mktemp)"
  crontab -l 2>/dev/null | \
    grep -Fv 'codex-candidate-step100-canary-monitor' | \
    grep -Fv 'codex-candidate-repeated-tblite-monitor' >"$tmp" || true
  printf '*/2 * * * * %s --root %s --health-log %s --step %s # codex-candidate-step100-canary-monitor\n' \
    "$SCRIPTS/monitor_candidate_step100_canaries.py" "$XROOT" \
    "$XROOT/next-sft-candidates-v1/monitor/candidate-step${step}-canaries-health.jsonl" \
    "$step" >>"$tmp"
  crontab "$tmp"
  rm -f "$tmp"
}

install_repeated_monitor() {
  local step="$1"
  local tmp
  tmp="$(mktemp)"
  crontab -l 2>/dev/null | \
    grep -Fv 'codex-candidate-step100-canary-monitor' | \
    grep -Fv 'codex-candidate-repeated-tblite-monitor' >"$tmp" || true
  printf '*/2 * * * * %s --root %s --health-log %s --step %s # codex-candidate-repeated-tblite-monitor\n' \
    "$SCRIPTS/monitor_candidate_repeated_tblite.py" "$XROOT" \
    "$XROOT/next-sft-candidates-v1/monitor/candidate-step${step}-repeated-tblite-health.jsonl" \
    "$step" >>"$tmp"
  crontab "$tmp"
  rm -f "$tmp"
}

launch_repeated() {
  local step="$1"
  local session="candidate-step${step}-repeated-tblite"
  local marker="$XROOT/candidate-step${step}-repeated-tblite.launched"
  test ! -e "$marker" || return 0
  tmux has-session -t "qwen35-4b-candidate-step${step}-seeds-serve"
  tmux new-session -d -s "$session" \
    "STEP='$step' bash '$SCRIPTS/run_candidate_repeated_tblite_mainbox.sh' 2>&1 | tee '$XROOT/logs/candidate-step${step}-repeated-tblite.launch.log'"
  install_repeated_monitor "$step"
  sleep 35
  tmux has-session -t "$session"
  touch "$marker"
  printf '%s launched repeated full100 TBLite step=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$step" >>"$LOG"
}

launch_canary() {
  local step="$1"
  local prior_step="$2"
  local marker="$XROOT/candidate-step${step}-two-seed-canaries.launched"
  test ! -e "$marker" || return 0
  test ! -e "$XROOT/candidate-step${step}-two-seed-canaries.preflight-failed"
  ! tmux has-session -t "candidate-step${prior_step}-two-seed-canaries" 2>/dev/null

  if tmux has-session -t "qwen35-4b-candidate-step${prior_step}-seeds-serve" 2>/dev/null; then
    tmux send-keys -t "qwen35-4b-candidate-step${prior_step}-seeds-serve" C-c
    sleep 20
    tmux kill-session -t "qwen35-4b-candidate-step${prior_step}-seeds-serve" 2>/dev/null || true
  fi
  for gpu in 0 1; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
    test "$used" -le 8192
  done

  tmux new-session -d -s "candidate-step${step}-two-seed-canaries" \
    "STEP='$step' bash '$SCRIPTS/run_candidate_step100_seed_canaries_mainbox.sh' 2>&1 | tee '$XROOT/logs/candidate-step${step}-two-seed-canaries.launch.log'"
  install_canary_monitor "$step"
  sleep 35
  tmux has-session -t "candidate-step${step}-two-seed-canaries"
  touch "$marker"
  printf '%s launched two-seed step%s held-out canaries after step%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$step" "$prior_step" >>"$LOG"
}

if test -e "$XROOT/candidate-step25-two-seed-canaries.complete"; then
  gate="$XROOT/candidate-step25-two-seed-canary-gate.json"
  test -s "$gate"
  if test "$(credible "$gate")" = 1; then
    launch_repeated 25
  else
    touch "$XROOT/candidate-checkpoint-sweep-no-credible-canary.marker"
    printf '%s steps 25, 50, 100, and 200 all failed the two-seed canary gate\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG"
  fi
  exit 0
fi

if test -e "$XROOT/candidate-step50-two-seed-canaries.complete"; then
  gate="$XROOT/candidate-step50-two-seed-canary-gate.json"
  test -s "$gate"
  if test "$(credible "$gate")" = 1; then
    launch_repeated 50
  else
    launch_canary 25 50
  fi
  exit 0
fi

if test -e "$XROOT/candidate-step200-two-seed-canaries.complete"; then
  gate="$XROOT/candidate-step200-two-seed-canary-gate.json"
  test -s "$gate"
  if test "$(credible "$gate")" = 1; then
    launch_repeated 200
  else
    touch "$XROOT/candidate-step200-no-credible-canary.marker"
    launch_canary 50 200
  fi
  exit 0
fi

if test -e "$XROOT/candidate-step100-two-seed-canaries.complete"; then
  gate="$XROOT/candidate-step100-two-seed-canary-gate.json"
  test -s "$gate"
  if test "$(credible "$gate")" = 1; then
    launch_repeated 100
  else
    launch_canary 200 100
  fi
  exit 0
fi

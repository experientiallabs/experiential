#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
AXO_ROOT="${AXO_ROOT:-${XROOT}/axolotl-sft}"
EVAL_OUT="${EVAL_OUT:?EVAL_OUT is required}"
EVAL_SESSION="${EVAL_SESSION:?EVAL_SESSION is required}"
SERVER_SESSION="${SERVER_SESSION:?SERVER_SESSION is required}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${AXO_ROOT}/scripts/serve_verified17_4b.sh}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${AXO_ROOT}/scripts/run_realverified14_sft.sh}"
TRAIN_SESSION="${TRAIN_SESSION:?TRAIN_SESSION is required}"
GPU_INDEX="${GPU_INDEX:?GPU_INDEX is required}"
SEED="${SEED:?SEED is required}"
RUN_SUFFIX="${RUN_SUFFIX:?RUN_SUFFIX is required}"
CONFIG="${CONFIG:-${AXO_ROOT}/configs/qwen35_4b_base_headroom4_sft.yaml}"
DATASET="${DATASET:-${AXO_ROOT}/sft-data/base-headroom4.qwen-materialized.jsonl}"
MANIFEST="${MANIFEST:-${AXO_ROOT}/sft-data/base-headroom4.manifest.json}"
OUTPUT="${OUTPUT:-${AXO_ROOT}/checkpoints/qwen35-4b-glm52-base-headroom4-sft-lr1e5-r64-${RUN_SUFFIX}}"
LOG="${LOG:-${AXO_ROOT}/logs/qwen35-4b-glm52-base-headroom4-sft-lr1e5-r64-${RUN_SUFFIX}.log}"
WANDB_NAME="${WANDB_NAME:-qwen35-4b-glm52-base-headroom4-sft-lr1e5-r64-${RUN_SUFFIX}}"
PREPARED_CACHE="${PREPARED_CACHE:-${AXO_ROOT}/cache/base-headroom4-${RUN_SUFFIX}}"

for required in "${SERVER_SCRIPT}" "${TRAIN_SCRIPT}" "${CONFIG}" "${DATASET}" "${MANIFEST}"; do
  test -r "${required}" || { echo "missing required file: ${required}" >&2; exit 1; }
done
test ! -e "${OUTPUT}" || { echo "refusing existing output: ${OUTPUT}" >&2; exit 1; }
test ! -e "${LOG}" || { echo "refusing existing log: ${LOG}" >&2; exit 1; }
tmux has-session -t "${EVAL_SESSION}" 2>/dev/null || {
  test -f "${EVAL_OUT}/summary.json" || {
    echo "evaluation session is absent without a complete summary" >&2
    exit 1
  }
}
tmux has-session -t "${SERVER_SESSION}" 2>/dev/null || {
  echo "owned server session is absent before handoff" >&2
  exit 1
}
tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null && {
  echo "training session already exists: ${TRAIN_SESSION}" >&2
  exit 1
}

while ! test -f "${EVAL_OUT}/summary.json"; do
  if ! tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; then
    echo "evaluation exited without summary; preserving server and diagnostics" >&2
    exit 1
  fi
  sleep 30
done

python3 - "${EVAL_OUT}/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if summary.get("attempted") != 14 or summary.get("scored") != 14:
    raise SystemExit(f"incomplete summary: {summary}")
if summary.get("status_counts") != {"scored": 14}:
    raise SystemExit(f"non-scored outcomes in summary: {summary}")
print(json.dumps(summary, sort_keys=True))
PY

for _ in $(seq 1 20); do
  tmux has-session -t "${EVAL_SESSION}" 2>/dev/null || break
  sleep 3
done
tmux has-session -t "${EVAL_SESSION}" 2>/dev/null && {
  echo "evaluation session remained alive after summary; refusing handoff" >&2
  exit 1
}

SESSION="${SERVER_SESSION}" "${SERVER_SCRIPT}" stop

for _ in $(seq 1 30); do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_INDEX}")"
  test "${used}" -le 8192 && break
  sleep 5
done
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_INDEX}")"
test "${used}" -le 8192 || {
  echo "GPU ${GPU_INDEX} still holds ${used} MiB after owned server shutdown" >&2
  exit 1
}

command="CONFIG='${CONFIG}' DATASET='${DATASET}' MANIFEST='${MANIFEST}' GPU_INDEX='${GPU_INDEX}' SEED='${SEED}' RUN_SUFFIX='${RUN_SUFFIX}' OUTPUT='${OUTPUT}' LOG='${LOG}' WANDB_NAME='${WANDB_NAME}' PREPARED_CACHE='${PREPARED_CACHE}' bash '${TRAIN_SCRIPT}'"
tmux new-session -d -s "${TRAIN_SESSION}" "${command}"
sleep 35
tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null || {
  tail -n 120 "${LOG}" >&2 || true
  echo "training session died during the 35-second liveness gate" >&2
  exit 1
}
echo "launched ${TRAIN_SESSION} on GPU ${GPU_INDEX}; log ${LOG}"

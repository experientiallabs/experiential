#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
VLLM_PY="${VLLM_PY:-/mnt/cache/xp/venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-${XROOT}/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${XROOT}/axolotl-sft/checkpoints/qwen35-4b-glm52-verified17-sft-lr1e5-r64-run1}"
ASSERT_ROUTING="${ASSERT_ROUTING:-${XROOT}/eval-code-b0d0568/xtoken-ops/offline/tb2/assert_adapter_routing.py}"
STEPS="${STEPS:-25 50 75 100}"
BASE_SERVED="${BASE_SERVED:-qwen35-4b-base}"
SESSION="${SESSION:-qwen35-4b-verified17-sft-sweep-serve}"
PORT="${PORT:-8120}"
API_BASE="http://127.0.0.1:${PORT}/v1"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
DP_SIZE="${DP_SIZE:-2}"
MAXLEN="${MAXLEN:-81920}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_UTIL="${GPU_UTIL:-0.85}"
LOG_DIR="${LOG_DIR:-${XROOT}/axolotl-sft/eval-logs}"
RUNTIME_DIR="${RUNTIME_DIR:-${XROOT}/axolotl-sft/eval-runtime}"
HF_HOME="${HF_HOME:-${XROOT}/hf}"
TMPDIR="${TMPDIR:-/mnt/datasets/tb2tmp}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XROOT}/axolotl-sft/cache/triton}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${XROOT}/axolotl-sft/cache/xdg}"
LOG="${LOG_DIR}/tb2-4b-vllm-sweep-tools.log"
MANIFEST="${RUNTIME_DIR}/tb2-4b-sweep-serve-manifest.json"
ACTION="${1:-start}"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}" "${HF_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

LORA_ARGS=()
adapter_count=0
for step in ${STEPS}; do
  adapter_dir="${ADAPTER_ROOT}/checkpoint-${step}"
  adapter_served="qwen35-4b-glm52-verified17-sft-step${step}"
  test -f "${adapter_dir}/adapter_model.safetensors" || {
    echo "missing adapter: ${adapter_dir}" >&2
    exit 1
  }
  LORA_ARGS+=("${adapter_served}=${adapter_dir}")
  adapter_count=$((adapter_count + 1))
done

VLLM_ARGV=(
  "${VLLM_PY}" -m vllm.entrypoints.openai.api_server
  --model "${BASE_MODEL}"
  --served-model-name "${BASE_SERVED}"
  --host 127.0.0.1
  --port "${PORT}"
  --tensor-parallel-size 1
  --data-parallel-size "${DP_SIZE}"
  --max-model-len "${MAXLEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --dtype bfloat16
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  --language-model-only
  --skip-mm-profiling
  --enable-prefix-caching
  --enable-lora
  --max-lora-rank 64
  --max-loras "${adapter_count}"
  --max-cpu-loras "${adapter_count}"
  --lora-modules "${LORA_ARGS[@]}"
)

case "${ACTION}" in
  args)
    printf '%q ' "${VLLM_ARGV[@]}"
    echo
    exit 0
    ;;
  stop)
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      tmux send-keys -t "${SESSION}" C-c
      sleep 20
      tmux kill-session -t "${SESSION}" 2>/dev/null || true
    fi
    exit 0
    ;;
  start) ;;
  *) echo "usage: $0 {start|args|stop}" >&2; exit 1 ;;
esac

test -x "${VLLM_PY}" || { echo "missing vLLM Python: ${VLLM_PY}" >&2; exit 1; }
test -d "${BASE_MODEL}" || { echo "missing base model: ${BASE_MODEL}" >&2; exit 1; }
test -r "${ASSERT_ROUTING}" || { echo "missing routing assertion: ${ASSERT_ROUTING}" >&2; exit 1; }
test "$(printf '%s' "${CUDA_DEVICES}" | awk -F, '{print NF}')" -eq "${DP_SIZE}" || {
  echo "CUDA device count does not match DP size" >&2
  exit 1
}
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "session exists: ${SESSION}" >&2; exit 1; }
curl -sf -m 5 "${API_BASE}/models" >/dev/null 2>&1 && { echo "port ${PORT} is busy" >&2; exit 1; }

for gpu in ${CUDA_DEVICES//,/ }; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")"
  test "${used}" -le 8192 || { echo "GPU ${gpu} is busy with ${used} MiB" >&2; exit 1; }
done

"${VLLM_PY}" - "${MANIFEST}" "${BASE_MODEL}" "${BASE_SERVED}" "${API_BASE}" "${CUDA_DEVICES}" "${DP_SIZE}" "${MAXLEN}" "${MAX_NUM_SEQS}" "${GPU_UTIL}" "${ADAPTER_ROOT}" ${STEPS} <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest, base, base_served, api_base, devices, dp, maxlen, maxseqs, gpu_util, root, *steps = sys.argv[1:]
adapters = []
for step in steps:
    path = Path(root) / f"checkpoint-{step}" / "adapter_model.safetensors"
    adapters.append({
        "step": int(step),
        "served_model": f"qwen35-4b-glm52-verified17-sft-step{step}",
        "adapter_dir": str(path.parent),
        "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
payload = {
    "base_snapshot": base,
    "base_served": base_served,
    "api_base": api_base,
    "cuda_visible_devices": devices,
    "tensor_parallel_size": 1,
    "data_parallel_size": int(dp),
    "max_model_len": int(maxlen),
    "max_num_seqs": int(maxseqs),
    "gpu_memory_utilization": float(gpu_util),
    "enable_lora": True,
    "max_lora_rank": 64,
    "enable_auto_tool_choice": True,
    "tool_call_parser": "qwen3_xml",
    "reasoning_parser": None,
    "adapters": adapters,
}
Path(manifest).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

command="export PATH='/mnt/cache/xp/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin' CUDA_HOME='/usr/local/cuda' VLLM_USE_DEEP_GEMM=0 CUDA_VISIBLE_DEVICES='${CUDA_DEVICES}' HF_HOME='${HF_HOME}' TMPDIR='${TMPDIR}' TRITON_CACHE_DIR='${TRITON_CACHE_DIR}' XDG_CACHE_HOME='${XDG_CACHE_HOME}'; $(printf '%q ' "${VLLM_ARGV[@]}") 2>&1 | tee '${LOG}'"
tmux new-session -d -s "${SESSION}" "${command}"
echo "serving in ${SESSION}; log ${LOG}"

for _ in $(seq 1 150); do
  if curl -sf -m 5 "${API_BASE}/models" >/dev/null 2>&1; then
    for step in ${STEPS}; do
      "${VLLM_PY}" "${ASSERT_ROUTING}" \
        --api-base "${API_BASE}" \
        --base-model "${BASE_SERVED}" \
        --adapter-model "qwen35-4b-glm52-verified17-sft-step${step}"
    done
    sha256sum "${MANIFEST}"
    exit 0
  fi
  tmux has-session -t "${SESSION}" 2>/dev/null || {
    tail -80 "${LOG}" >&2 || true
    echo "vLLM exited during startup" >&2
    exit 1
  }
  sleep 10
done
echo "timed out waiting for ${API_BASE}/models" >&2
exit 1

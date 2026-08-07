#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
AXO_ROOT="${AXO_ROOT:-${XROOT}/axolotl-sft}"
VENV="${VENV:-${AXO_ROOT}/venv}"
CONFIG="${CONFIG:-${AXO_ROOT}/configs/qwen35_4b_realverified14_sft.yaml}"
DATASET="${DATASET:-${AXO_ROOT}/sft-data/real-verifier-perfect14.qwen-materialized.jsonl}"
MANIFEST="${MANIFEST:-${AXO_ROOT}/sft-data/real-verifier-perfect14.manifest.json}"
GPU_INDEX="${GPU_INDEX:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-run1}"
SEED="${SEED:-20260807}"
OUTPUT="${OUTPUT:-${AXO_ROOT}/checkpoints/qwen35-4b-glm52-realverified14-sft-lr1e5-r64-${RUN_SUFFIX}}"
LOG="${LOG:-${AXO_ROOT}/logs/qwen35-4b-glm52-realverified14-sft-lr1e5-r64-${RUN_SUFFIX}.log}"
WANDB_NAME="${WANDB_NAME:-qwen35-4b-glm52-realverified14-sft-lr1e5-r64-${RUN_SUFFIX}}"
PREPARED_CACHE="${PREPARED_CACHE:-${AXO_ROOT}/cache/realverified14-${RUN_SUFFIX}}"

for required in "${VENV}/bin/axolotl" "${VENV}/bin/accelerate" "${CONFIG}" "${DATASET}" "${MANIFEST}"; do
  test -r "${required}" || { echo "missing required file: ${required}" >&2; exit 1; }
done
test ! -e "${OUTPUT}" || { echo "refusing to overwrite: ${OUTPUT}" >&2; exit 1; }
test ! -e "${LOG}" || { echo "refusing to overwrite: ${LOG}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PATH="${VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export HF_HOME="${XROOT}/hf"
export TMPDIR="/mnt/datasets/tb2tmp"
export TRITON_CACHE_DIR="${AXO_ROOT}/cache/triton"
export XDG_CACHE_HOME="${AXO_ROOT}/cache/xdg"
export WANDB_PROJECT=x_token
export WANDB_NAME
mkdir -p "$(dirname "${LOG}")" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

sha256sum "${CONFIG}" "${DATASET}" "${MANIFEST}"
df -h /mnt/datasets
test "$(command -v accelerate)" = "${VENV}/bin/accelerate" || {
  echo "accelerate resolved outside the Axolotl venv: $(command -v accelerate)" >&2
  exit 1
}
"${VENV}/bin/python" -c 'import axolotl, torch; print(axolotl.__version__, torch.__version__, torch.cuda.get_device_name(0))'

"${VENV}/bin/python" -m axolotl.cli.train "${CONFIG}" \
  --output_dir="${OUTPUT}" \
  --dataset_prepared_path="${PREPARED_CACHE}" \
  --seed="${SEED}" \
  --wandb_name="${WANDB_NAME}" \
  2>&1 | tee "${LOG}"

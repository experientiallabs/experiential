#!/usr/bin/env bash
set -euo pipefail

experiment_root=/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200/axolotl-sft
uv_bin=/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200/bin/uv

export UV_CACHE_DIR="$experiment_root/cache/uv"
export HF_HOME=/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200/hf
export TMPDIR="$experiment_root/cache/tmp"
export TRITON_CACHE_DIR="$experiment_root/cache/triton"
export UV_TORCH_BACKEND=cu130

mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$TRITON_CACHE_DIR"

"$uv_bin" venv --python /usr/bin/python3.12 "$experiment_root/venv"
"$uv_bin" pip install \
  --python "$experiment_root/venv/bin/python" \
  torch==2.12.0 \
  torchvision
"$uv_bin" pip install \
  --python "$experiment_root/venv/bin/python" \
  --no-build-isolation \
  'axolotl[deepspeed]==0.17.0'

"$experiment_root/venv/bin/python" -c \
  'import axolotl, torch; print(f"axolotl={axolotl.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'

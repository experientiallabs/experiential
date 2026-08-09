#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
VENV="$AXO_ROOT/venv"
CONFIG="$AXO_ROOT/configs/qwen35_4b_candidate_realverified_sft_mainbox.yaml"
DATA_ROOT="$XROOT/next-sft-candidates-v1/sft-data"
DATASET="$DATA_ROOT/candidate192.realverified-perfect.qwen-materialized.jsonl"
MANIFEST="$DATA_ROOT/candidate192.realverified-perfect.manifest.json"
MASK_AUDIT="$DATA_ROOT/candidate192.realverified-perfect.mask-audit.json"

if test "$#" -ne 2; then
  echo "usage: $0 GPU_INDEX SEED" >&2
  exit 2
fi
GPU_INDEX="$1"
SEED="$2"
RUN_NAME="qwen35-4b-glm52-candidate-realverified-sft-lr1e5-r64-seed${SEED}"
OUTPUT="$AXO_ROOT/checkpoints/$RUN_NAME"
LOG="$AXO_ROOT/logs/$RUN_NAME.log"
PREPARED_CACHE="$AXO_ROOT/cache/candidate-realverified-seed${SEED}"

for required in "$VENV/bin/axolotl" "$VENV/bin/accelerate" "$CONFIG" \
  "$DATASET" "$MANIFEST" "$MASK_AUDIT"; do
  test -r "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
test "$(jq -r .selected_rows "$MANIFEST")" -ge 50
test "$(jq -r .totals.rows "$MASK_AUDIT")" = "$(jq -r .selected_rows "$MANIFEST")"
test ! -e "$OUTPUT" || { echo "refusing to overwrite: $OUTPUT" >&2; exit 1; }
test ! -e "$LOG" || { echo "refusing to overwrite: $LOG" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PATH="$VENV/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export HF_HOME=/scratch/xtoken-offline-9b-20260727/cache/huggingface
export TMPDIR=/scratch/tb2tmp
export TRITON_CACHE_DIR="$AXO_ROOT/cache/triton"
export XDG_CACHE_HOME="$AXO_ROOT/cache/xdg"
export WANDB_PROJECT=x_token
export WANDB_NAME="$RUN_NAME"
mkdir -p "$(dirname "$LOG")" "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

sha256sum "$CONFIG" "$DATASET" "$MANIFEST" "$MASK_AUDIT"
"$VENV/bin/python" -c 'import axolotl, torch; print(axolotl.__version__, torch.__version__, torch.cuda.get_device_name(0))'
"$VENV/bin/python" -m axolotl.cli.train "$CONFIG" \
  --output_dir="$OUTPUT" \
  --dataset_prepared_path="$PREPARED_CACHE" \
  --seed="$SEED" \
  --wandb_name="$RUN_NAME" \
  2>&1 | tee "$LOG"

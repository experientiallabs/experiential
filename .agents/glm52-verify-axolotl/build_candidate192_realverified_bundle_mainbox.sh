#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
DATA_ROOT="$XROOT/next-sft-candidates-v1"
SFT_ROOT="$DATA_ROOT/sft-data"
PYTHON=/scratch/xtoken-offline-9b-20260727/.venv/bin/python
AXO_PYTHON="$AXO_ROOT/venv/bin/python"
REPLAY_ROOT="$DATA_ROOT/teacher-replay/run1"
INPUT_AUDIT="$SFT_ROOT/candidate192.audit.jsonl"
INPUT_QWEN="$SFT_ROOT/candidate192.qwen-materialized.jsonl"
OUTPUT_AUDIT="$SFT_ROOT/candidate192.realverified-perfect.audit.jsonl"
OUTPUT_QWEN="$SFT_ROOT/candidate192.realverified-perfect.qwen-materialized.jsonl"
OUTPUT_LEDGER="$SFT_ROOT/candidate192.realverified-perfect.ledger.jsonl"
OUTPUT_MANIFEST="$SFT_ROOT/candidate192.realverified-perfect.manifest.json"
OUTPUT_MASK_AUDIT="$SFT_ROOT/candidate192.realverified-perfect.mask-audit.json"
MODEL=Qwen/Qwen3.5-4B
REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

for required in "$PYTHON" "$AXO_PYTHON" "$INPUT_AUDIT" "$INPUT_QWEN" \
  "$REPLAY_ROOT/summary.json" "$AXO_ROOT/scripts/build_real_verifier_filtered_sft.py" \
  "$AXO_ROOT/scripts/audit_axolotl_masks.py"; do
  test -r "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
for output in "$OUTPUT_AUDIT" "$OUTPUT_QWEN" "$OUTPUT_LEDGER" \
  "$OUTPUT_MANIFEST" "$OUTPUT_MASK_AUDIT"; do
  test ! -e "$output" || { echo "refusing to overwrite: $output" >&2; exit 1; }
done

"$PYTHON" "$AXO_ROOT/scripts/build_real_verifier_filtered_sft.py" \
  --audit-dataset "$INPUT_AUDIT" \
  --qwen-dataset "$INPUT_QWEN" \
  --replay-root "$REPLAY_ROOT/episodes" \
  --replay-summary "$REPLAY_ROOT/summary.json" \
  --minimum-reward 1.0 \
  --output-audit "$OUTPUT_AUDIT" \
  --output-qwen "$OUTPUT_QWEN" \
  --output-ledger "$OUTPUT_LEDGER" \
  --output-manifest "$OUTPUT_MANIFEST"

SELECTED_ROWS="$(jq -r .selected_rows "$OUTPUT_MANIFEST")"
test "$SELECTED_ROWS" -ge 50 || {
  echo "real-verifier set has only $SELECTED_ROWS rows; require at least 50" >&2
  exit 1
}
export HF_HOME=/scratch/xtoken-offline-9b-20260727/cache/huggingface
"$AXO_PYTHON" "$AXO_ROOT/scripts/audit_axolotl_masks.py" \
  --dataset "$OUTPUT_QWEN" --model "$MODEL" --revision "$REVISION" \
  --sequence-len 32768 --output "$OUTPUT_MASK_AUDIT"
sha256sum "$OUTPUT_AUDIT" "$OUTPUT_QWEN" "$OUTPUT_LEDGER" \
  "$OUTPUT_MANIFEST" "$OUTPUT_MASK_AUDIT"

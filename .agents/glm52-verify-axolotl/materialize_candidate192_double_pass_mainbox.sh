#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
DATA_ROOT="$XROOT/next-sft-candidates-v1"
SFT_ROOT="$DATA_ROOT/sft-data"
CORPUS=/scratch/xtoken-offline-9b-20260727/tmax/offline-c43abfc-20260802/corpus-40960.jsonl
PRIMARY="$DATA_ROOT/judgments/sonnet46.jsonl"
ADJUDICATOR="$DATA_ROOT/judgments/opus45.jsonl"
PYTHON=/scratch/xtoken-offline-9b-20260727/.venv/bin/python
AXO_PYTHON="$AXO_ROOT/venv/bin/python"
MODEL=Qwen/Qwen3.5-4B
REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

RAW_QWEN="$SFT_ROOT/candidate192.double-pass.jsonl"
RAW_SUMMARY="$SFT_ROOT/candidate192.double-pass.summary.json"
LEDGER="$SFT_ROOT/candidate192.admission-ledger.jsonl"
AUDIT="$SFT_ROOT/candidate192.audit.jsonl"
AUDIT_MANIFEST="$SFT_ROOT/candidate192.audit-manifest.json"
MATERIALIZED="$SFT_ROOT/candidate192.qwen-materialized.jsonl"
MATERIALIZED_SUMMARY="$SFT_ROOT/candidate192.qwen-materialized.summary.json"
MASK_AUDIT="$SFT_ROOT/candidate192.qwen-materialized.mask-audit.json"

for required in "$CORPUS" "$PRIMARY" "$ADJUDICATOR" \
  "$AXO_ROOT/scripts/build_sft_subset.py" \
  "$AXO_ROOT/scripts/build_verified_audit_dataset.py" \
  "$AXO_ROOT/scripts/materialize_qwen_messages.py" \
  "$AXO_ROOT/scripts/audit_axolotl_masks.py" "$PYTHON" "$AXO_PYTHON"; do
  test -r "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
test "$(wc -l < "$PRIMARY")" -eq 192
test "$(wc -l < "$ADJUDICATOR")" -eq 192
test "$(jq -r 'select(.decision == null) | .row_index' "$PRIMARY" | wc -l)" -eq 0
test "$(jq -r 'select(.decision == null) | .row_index' "$ADJUDICATOR" | wc -l)" -eq 0
for output in "$RAW_QWEN" "$RAW_SUMMARY" "$LEDGER" "$AUDIT" \
  "$AUDIT_MANIFEST" "$MATERIALIZED" "$MATERIALIZED_SUMMARY" "$MASK_AUDIT"; do
  test ! -e "$output" || { echo "refusing to overwrite: $output" >&2; exit 1; }
done
mkdir -p "$SFT_ROOT"

"$PYTHON" "$AXO_ROOT/scripts/build_sft_subset.py" \
  --corpus "$CORPUS" --primary "$PRIMARY" --adjudicator "$ADJUDICATOR" \
  --output "$RAW_QWEN" --summary "$RAW_SUMMARY"

"$PYTHON" "$AXO_ROOT/scripts/build_verified_audit_dataset.py" \
  --corpus "$CORPUS" --primary "$PRIMARY" --adjudicator "$ADJUDICATOR" \
  --training-view "$RAW_QWEN" --ledger "$LEDGER" --admitted "$AUDIT" \
  --manifest "$AUDIT_MANIFEST"

export HF_HOME=/scratch/xtoken-offline-9b-20260727/cache/huggingface
"$AXO_PYTHON" "$AXO_ROOT/scripts/materialize_qwen_messages.py" \
  --input "$RAW_QWEN" --output "$MATERIALIZED" \
  --summary "$MATERIALIZED_SUMMARY" --model "$MODEL" --revision "$REVISION"

"$AXO_PYTHON" "$AXO_ROOT/scripts/audit_axolotl_masks.py" \
  --dataset "$MATERIALIZED" --model "$MODEL" --revision "$REVISION" \
  --sequence-len 32768 --output "$MASK_AUDIT"

sha256sum "$RAW_QWEN" "$LEDGER" "$AUDIT" "$MATERIALIZED" "$MASK_AUDIT"

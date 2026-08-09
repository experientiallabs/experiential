#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
CORPUS=/scratch/xtoken-offline-9b-20260727/tmax/offline-c43abfc-20260802/corpus-40960.jsonl
INDICES="$XROOT/next-sft-candidates-v1/candidate192.jsonl"
SCRIPT="$XROOT/axolotl-sft/scripts/judge_trajectories.py"
PYTHON=/scratch/xtoken-offline-9b-20260727/.venv/bin/python
CORPUS_SHA=7220f5d58e41933e38c46a29eee37ff4da4a21e8901ea27f9ead624cc6df911a
INDICES_SHA=4e3ce23d213ae6f1a06ccf519dfde0d10028d22741eeee34ce41eba9b26e3a05
REVISION=c43abfc846e8dffc5d9b684614724f2471ea47dd-corpus-40960

if test "$#" -ne 4; then
  echo "usage: $0 MODEL_ID OUTPUT CONCURRENCY LOG" >&2
  exit 2
fi
MODEL_ID="$1"
OUTPUT="$2"
CONCURRENCY="$3"
LOG="$4"

test -x "$PYTHON"
test -r "$SCRIPT"
test "$(sha256sum "$CORPUS" | cut -d' ' -f1)" = "$CORPUS_SHA"
test "$(sha256sum "$INDICES" | cut -d' ' -f1)" = "$INDICES_SHA"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
export AWS_PROFILE=claas-bedrock

"$PYTHON" "$SCRIPT" \
  --corpus "$CORPUS" \
  --indices "$INDICES" \
  --output "$OUTPUT" \
  --model-id "$MODEL_ID" \
  --region us-west-1 \
  --corpus-revision "$REVISION" \
  --source-sha256 "$CORPUS_SHA" \
  --concurrency "$CONCURRENCY" \
  --max-attempts 5 \
  2>&1 | tee -a "$LOG"

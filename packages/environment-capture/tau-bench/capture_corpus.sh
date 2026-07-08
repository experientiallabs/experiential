#!/usr/bin/env bash
# Capture a multi-domain tau2-bench trace corpus across airline + retail + telecom, then convert and
# merge into one wmh OTel corpus. This is how `packages/environment-capture/tau-bench/traces.otel.jsonl` was grown to
# ~1000 traces for the trace-scaling-law experiment (docs/trace_scaling.md).
#
#   packages/environment-capture/tau-bench/capture_corpus.sh
#
# Distinct-task budget per domain (1 trial each, so every trace is a different task — no repeats):
#   airline   50  (all tasks)
#   retail   114  (all tasks)
#   telecom  ~880 (of 2285 in the `full` split) -> ~1000 distinct total
# Reward < 1.0 sims are KEPT (the recorded tool results are real either way; reward rides along in
# trace metadata for later filtering). Only infrastructure errors and tool-call-free chats drop out.
#
# NOTE: telecom on a single Opus model throttles hard on Bedrock (see capture_telecom_multimodel.py,
# which shards telecom across 4.6/4.7/4.8 and is how the committed telecom traces were captured).
# This script's telecom leg is the simple single-model path; prefer the multimodel one at scale.
#
# Resumable: a domain whose results.json already exists is skipped, so a re-run only fills gaps.
# Live on Bedrock (the default AWS profile has Bedrock access).
set -euo pipefail
cd "$(dirname "$0")"

# Concurrency is deliberately LOW (3): Opus 4.8 on Bedrock throttles
# (litellm ServiceUnavailableError) under sustained parallel load, and telecom's longer trajectories
# make more LLM calls per task — at concurrency 8 ~80% of telecom sims failed permanently. Lower
# concurrency + task-level retries trades wall-clock for a reliable yield.
CONCURRENCY="${CONCURRENCY:-3}"
MAX_RETRIES="${MAX_RETRIES:-5}"     # tau2 task-level retries for failed sims (default 3)
RETRY_DELAY="${RETRY_DELAY:-5.0}"   # seconds between task retries (default 1.0)
AGENT_LLM="bedrock/us.anthropic.claude-opus-4-8"
USER_LLM="bedrock/us.anthropic.claude-opus-4-8"
OUT="${OUT:-./traces.otel.jsonl}"

export TAU2_DATA_DIR="$PWD/tau2-bench/data"
export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"

# domain : task-split : num-tasks. The default "base" split exposes only airline 50 / retail 114 /
# telecom 114; telecom's "full" split unlocks 2285 tasks, so we draw the bulk from there. num-tasks
# is over-budgeted (esp. telecom) to absorb sims that still fail after retries and chats with no tool
# call, and still clear ~1000 distinct: airline ~47 + retail ~74 + telecom ~880.
DOMAINS=("airline:base:50" "retail:base:114" "telecom:full:980")

# --auto-resume makes a re-run RETRY only the failed/missing tasks in an existing save dir (keeping
# the ones that already succeeded), so re-running this script tops up a throttled run rather than
# starting over or skipping wholesale. Idempotent: when a domain is fully captured, the resume is a
# no-op and it moves on.
run_domain() {
  local domain="$1" split="$2" n="$3" save="capture_${1}"
  echo "=== ${domain}: capturing/resuming ${n} tasks (split=${split}) @ concurrency ${CONCURRENCY} ==="
  .venv/bin/tau2 run \
    --domain "$domain" \
    --agent-llm "$AGENT_LLM" --agent-llm-args '{}' \
    --user-llm  "$USER_LLM"  --user-llm-args '{}' \
    --task-split-name "$split" \
    --num-trials 1 --num-tasks "$n" --max-concurrency "$CONCURRENCY" \
    --max-retries "$MAX_RETRIES" --retry-delay "$RETRY_DELAY" \
    --auto-resume \
    --save-to "$save"
}

for spec in "${DOMAINS[@]}"; do
  IFS=":" read -r d s n <<< "$spec"
  run_domain "$d" "$s" "$n"
done

# Convert each domain's results into its own OTel shard, then concatenate into the merged corpus.
# (convert_to_wmh.py infers a single domain per results.json, so convert per-domain and cat.)
echo "=== converting + merging -> ${OUT} ==="
: > "$OUT"
total=0
for spec in "${DOMAINS[@]}"; do
  domain="${spec%%:*}"
  res="tau2-bench/data/simulations/capture_${domain}/results.json"
  shard="/tmp/tau2_${domain}.otel.jsonl"
  .venv/bin/python convert_to_wmh.py "$res" --out "$shard" --benchmark tau2-bench
  cat "$shard" >> "$OUT"
done
# Report distinct trace count in the merged corpus.
total=$(python3 -c "
import json
ids=set()
for line in open('$OUT'):
    line=line.strip()
    if line: ids.add(json.loads(line)['traceId'])
print(len(ids))
")
echo "=== merged corpus: ${total} distinct traces -> ${OUT} ==="

#!/usr/bin/env bash
# GEPA value-per-dollar ROUND 2: E1 frontier direct comparison, E2 config-vs-executor,
# E3 spend axis (plan: .agents/docs/proposals/gepa-value-per-dollar.md, "Round 2").
# E4 (headroom probe) is a separate script: probe_gepa_headroom.py.
#
# Same harness as round 1: one metered invocation per arm, judge pinned Opus 4.8 rubric-v2,
# seeds 0,1, test-cap 40, deterministic #97 split. Cheapest arms first.
#
# Run:   AWS_PROFILE=default AWS_REGION=us-east-1 bash .agents/scripts/run_gepa_vpd_round2.sh
# Cost:  ~$255 total. Kill switches: E2 $30, E3 $90, E1 gepa $200.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=.agents/docs/research/gepa_vpd_results
mkdir -p "$OUT"
RUN="uv run python .agents/scripts/run_gepa_scaling.py"
HAIKU=us.anthropic.claude-haiku-4-5-20251001-v1:0
MINI=openai/gpt-5.4-mini
OPUS=us.anthropic.claude-opus-4-7
COMMON="--examples packages/environment-capture --counts 100000 --seeds 0,1 \
  --sample-turns sampled --test-cap 40 --concurrency 8"
WINNING="--budgets 8 --minibatch 8 --gepa-val-steps 90 --val-fill inclusive --recheck-steps 30"

run_cell() { # $1=tag $2=suite $3=opt-model $4...=extra args
  local tag=$1 suite=$2 model=$3; shift 3
  $RUN "$suite" $COMMON --opt-model "$model" "$@" \
    --out "$OUT/${tag}.json" 2>&1 | tee "$OUT/${tag}.log"
}

# E2: Mini x terminal at the TIER config (b=4, mb=3, val=24 greedy, no recheck).
# Round 1's winning-config lift on this cell was +0.051; this arm isolates the config's share.
run_cell e2_terminal_mini_gepa_tier terminal-tasks "$MINI" \
  --budgets 4 --minibatch 3 --gepa-val-steps 24 --val-fill greedy

# E3: Haiku x tau at DOUBLE budget (b=16), winning config otherwise. H4: does spend keep
# paying past b=8's 0.893 parity point?
run_cell e3_tau_haiku_gepa_b16 tau-bench "$HAIKU" \
  --budgets 16 --minibatch 8 --gepa-val-steps 90 --val-fill inclusive --recheck-steps 30

# E1: Opus 4.7 x tau, rag anchor + the SAME winning-config GEPA the cheap executors got.
# In-harness frontier lift (replaces the #97 cross-ref) + the parity stress test.
run_cell e1_tau_opus_rag       tau-bench "$OPUS" --budgets 0
run_cell e1_tau_opus_gepa_self tau-bench "$OPUS" $WINNING

echo "done -> $OUT"

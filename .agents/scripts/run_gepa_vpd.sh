#!/usr/bin/env bash
# GEPA value-per-dollar, targeted cheap-executor cells
# (plan: .agents/docs/proposals/gepa-value-per-dollar.md).
#
# 3 cells x (rag anchor + winning-config GEPA arms), one metered invocation per arm so each
# RunTracker total IS that arm's cost. Build $ per GEPA arm = arm total minus the same cell's
# _rag total (differences out the shared test-scoring overhead). Judge pinned Opus 4.8
# rubric-v2, seeds 0,1, same deterministic split as #97.
#
# Cells: T1 Haiku x terminal (old optimizer emitted base here; can the winning config unlock
# it?), T2 Haiku x tau (confirm grid +0.049), T3 GPT-5.4 Mini x terminal (replicate the grid
# headline +0.157; auto-skipped without OPENAI_API_KEY).
#
# The gepa-strong-reflect arms (reflection LM = Opus 4.7, executor stays cheap) need the
# --reflect-model knob (reflection_provider in wmh/optimize/gepa.py); uncomment once landed.
#
# Run:   AWS_PROFILE=default AWS_REGION=us-east-1 bash .agents/scripts/run_gepa_vpd.sh
# Cost:  ~$150-250 total. Kill any arm past $60 metered (plan's kill switch).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=.agents/docs/research/gepa_vpd_results
mkdir -p "$OUT"
RUN="uv run python .agents/scripts/run_gepa_scaling.py"
HAIKU=us.anthropic.claude-haiku-4-5-20251001-v1:0
MINI=openai/gpt-5.4-mini
COMMON="--examples packages/environment-capture --counts 100000 --seeds 0,1 \
  --sample-turns sampled --test-cap 40 --concurrency 8"
WINNING="--budgets 8 --minibatch 8 --gepa-val-steps 90 --val-fill inclusive --recheck-steps 30"

run_cell() { # $1=tag $2=suite $3=opt-model $4...=extra args
  local tag=$1 suite=$2 model=$3; shift 3
  $RUN "$suite" $COMMON --opt-model "$model" "$@" \
    --out "$OUT/${tag}.json" 2>&1 | tee "$OUT/${tag}.log"
}

# T1: Haiku 4.5 x terminal-tasks
run_cell t1_terminal_haiku_rag         terminal-tasks "$HAIKU" --budgets 0
run_cell t1_terminal_haiku_gepa_self   terminal-tasks "$HAIKU" $WINNING
run_cell t1_terminal_haiku_gepa_strong terminal-tasks "$HAIKU" $WINNING \
  --reflect-model us.anthropic.claude-opus-4-7

# T2: Haiku 4.5 x tau-bench (b=16 spend arm stays opt-in; run after the b=8 verdict)
run_cell t2_tau_haiku_rag              tau-bench "$HAIKU" --budgets 0
run_cell t2_tau_haiku_gepa_self        tau-bench "$HAIKU" $WINNING
run_cell t2_tau_haiku_gepa_strong      tau-bench "$HAIKU" $WINNING \
  --reflect-model us.anthropic.claude-opus-4-7
# run_cell t2_tau_haiku_gepa_b16       tau-bench "$HAIKU" --budgets 16 --minibatch 8 \
#   --gepa-val-steps 90 --val-fill inclusive --recheck-steps 30

# T3: GPT-5.4 Mini x terminal-tasks (skipped without an OpenAI key)
if [ -n "${OPENAI_API_KEY:-}" ]; then
  run_cell t3_terminal_mini_rag       terminal-tasks "$MINI" --budgets 0
  run_cell t3_terminal_mini_gepa_self terminal-tasks "$MINI" $WINNING
else
  echo "OPENAI_API_KEY absent: skipping T3 (GPT-5.4 Mini x terminal)"
fi

echo "done -> $OUT (fidelity in *.json, metered $ in each log's 'usage:' lines)"

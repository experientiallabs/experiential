#!/usr/bin/env bash
# One-command reproduction of the Bench-VERIFY scorecard (bird-sql, 40 balanced
# trajectories, GoldJudge opus-4-8, k=3 votes at temperature 0.7).
# Run from the repo root. Bedrock credentials in .env.
set -euo pipefail
uv run python packages/environment-capture/bird-sql/fetch_data.py
uv run python .agents/scripts/gev_bench/run_verify_bench.py \
  --model us.anthropic.claude-opus-4-8 --region us-east-1 \
  --k 3 --temperature 0.7 --n-per-class 20
echo "Bench-VERIFY reproduced -> .agents/docs/research/gev_bench_results/verify"

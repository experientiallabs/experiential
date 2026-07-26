#!/usr/bin/env bash
# One-command reproduction of the Bench-EXEC scorecard (bird-sql, 8 held-out
# scenarios x {haiku-4.5, opus-4.8} x k=2 x {real, sim} = 64 episodes, ~9 min).
# Run from the repo root. Bedrock credentials in .env.
set -euo pipefail
uv run python packages/environment-capture/bird-sql/fetch_data.py
uv run python .agents/scripts/gev_bench/exec_bench.py --scenarios 8 --k 2
echo "Bench-EXEC reproduced -> .agents/docs/research/gev_bench_results/exec"

#!/usr/bin/env bash
# One-command reproduction of the Bench-GEN scorecard (tau-bench, 100 traces, budget 15).
# Run from the repo root. Bedrock credentials in .env. ~6 min wall.
set -euo pipefail
OUT=.agents/docs/research/gev_bench_results/gen
CORPUS=packages/environment-capture/tau-bench/traces.otel.jsonl

uv run wmo scenarios build --file "$CORPUS" --limit 100 --budget 15 \
  --provider bedrock --model claude-opus-4-8 --region us-east-1 \
  --out "$OUT/tau_scenarios.json"

uv run wmo build --name gev-tau --no-interactive --file "$CORPUS" \
  --provider bedrock --model claude-haiku-4-5 --judge-model claude-haiku-4-5 \
  --region us-east-1 --fidelity low --root .wmo </dev/null

uv run python .agents/scripts/gev_bench/run_verify.py \
  --scenarios "$OUT/tau_scenarios.json" --file "$CORPUS" \
  --name gev-tau --max-steps 10 --out "$OUT/verification_report.json"

uv run python .agents/scripts/gev_bench/build_report.py \
  --scenarios "$OUT/tau_scenarios.json" --file "$CORPUS" \
  --report "$OUT/verification_report.json" --outdir "$OUT"

echo "Bench-GEN reproduced -> $OUT (blind labels are the manual leg: labels_template.jsonl)"

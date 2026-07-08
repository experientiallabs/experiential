#!/usr/bin/env bash
# Build a world model from Mastra AI-tracing spans, end to end.
#
# Mastra records agent runs as AI-tracing spans typed by `type` (model_generation / tool_call / ...).
# Pull them live from a running Mastra server, or ingest an exported spans file.
set -euo pipefail

MODEL="${1:-mastra-demo}"

# Option A — pull live from a running Mastra server (fetches {base}/api/observability/traces).
#   MASTRA_URL defaults to Mastra's dev server; --api-key is sent as a bearer token if needed.
MASTRA_URL="${MASTRA_URL:-http://localhost:4111}"
uv run wmh build --name "$MODEL" --source mastra --pull --project "$MASTRA_URL" --no-interactive

# Option B — build from an exported spans file instead (single span / array / {"spans": [...]} /
# {"traces": [...]} / JSONL):
#   uv run wmh build --name "$MODEL" --source mastra --file mastra_spans.json --no-interactive

echo "Built world model '$MODEL' from Mastra AI-tracing spans."

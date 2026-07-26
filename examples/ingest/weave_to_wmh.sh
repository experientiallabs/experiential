#!/usr/bin/env bash
# Ingest Weights & Biases Weave traces into the wmh build pipeline.
#
# Weave records agent executions as "Calls". Export them from the Weave UI
# (JSON or JSONL) or pull live via the service API.
#
# Usage (file):
#   wmh ingest run --source weave --file weave_calls.json
#   wmh ingest run --source weave --file weave_calls.jsonl
#
# Usage (live pull — requires WANDB_API_KEY):
#   export WANDB_API_KEY="your-wandb-api-key"
#   wmh ingest run --source weave --project "myteam/myproject" --limit 500
#
# Then build the world model:
#   wmh build --source weave --file weave_calls.json

set -euo pipefail

FILE="${1:?Usage: $0 <weave-export.json|.jsonl>}"

echo "→ Ingesting Weave traces from: $FILE"
wmh ingest run --source weave --file "$FILE"

echo "→ Building world model from ingested traces..."
wmh build --source weave --file "$FILE"

echo "✓ Done. Run 'wmh serve' to start the world model server."

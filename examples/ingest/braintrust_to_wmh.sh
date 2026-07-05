#!/usr/bin/env bash
# Ingest Braintrust traces into a world model, end to end.
#
# Braintrust logs SPANS AS ROWS in an experiment/project log (not OTLP spans). A trace is the set of
# rows sharing a `root_span_id`. Export the rows via the fetch API (or the SDK), then let the
# `braintrust` adapter normalize them into the OTel-GenAI JSONL that `wmh build` consumes.
set -euo pipefail

EXPORT="${1:-braintrust_export.json}"
MODEL="${2:-braintrust-demo}"

# 1) Export from Braintrust (pick one). The adapter accepts a single span row, a JSON array of rows,
#    an API page wrapper ({"events": [...]} or {"data": [...]}), or JSONL (one row per line).
#
#    Fetch API — project logs ({"events": [...]}); BRAINTRUST_API_KEY is your org key:
#      curl -s -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
#        "https://api.braintrust.dev/v1/project_logs/$PROJECT_ID/fetch" > "$EXPORT"
#
#    Fetch API — an experiment's spans:
#      curl -s -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
#        "https://api.braintrust.dev/v1/experiment/$EXPERIMENT_ID/fetch" > "$EXPORT"
#
#    SDK (Python): braintrust.api.* / dataset iteration -> dump each span row to JSON/JSONL.

# 2) Build directly from the Braintrust export — `--source braintrust` ingests it as part of build.
uv run wmh build --name "$MODEL" --source braintrust --file "$EXPORT" --no-interactive

echo "Built world model '$MODEL' from $EXPORT."

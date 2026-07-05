#!/usr/bin/env bash
# Ingest LangSmith runs into a world model, end to end.
#
# LangSmith (LangChain) exports a trace as a tree of RUNS (not OTLP spans). Pull the runs for a
# trace (or a project page) via the API or the SDK, then let the `langsmith` adapter normalize the
# run tree into the OTel-GenAI JSONL that `wmh build` consumes.
set -euo pipefail

EXPORT="${1:-langsmith_export.json}"
MODEL="${2:-langsmith-demo}"

# 1) Export from LangSmith (pick one). The adapter accepts a single run, a JSON array of runs, a
#    {"runs": [...]} wrapper, or JSONL (one run per line).
#
#    REST API — runs in a project (LANGCHAIN_API_KEY is your key; filter to one trace by id):
#      curl -s -X POST "${LANGCHAIN_ENDPOINT:-https://api.smith.langchain.com}/api/v1/runs/query" \
#        -H "x-api-key: $LANGCHAIN_API_KEY" -H "Content-Type: application/json" \
#        -d '{"session": ["<project-uuid>"], "limit": 100}' \
#        | python -c 'import sys,json; print(json.dumps(json.load(sys.stdin)["runs"]))' > "$EXPORT"
#
#    SDK (Python): dump runs to a JSON array (one trace, or a whole project):
#      uv run python - <<'PY' > "$EXPORT"
#      import json
#      from langsmith import Client
#      runs = Client().list_runs(project_name="my-project")  # or trace_id=<uuid>
#      json.dump([r.dict() for r in runs], sys.stdout, default=str)
#      PY

# 2) Build directly from the LangSmith export — `--source langsmith` ingests it as part of build.
uv run wmh build --name "$MODEL" --source langsmith --file "$EXPORT" --no-interactive

echo "Built world model '$MODEL' from $EXPORT."

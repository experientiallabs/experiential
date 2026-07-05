#!/usr/bin/env bash
# Ingest Arize Phoenix traces into a world model.
#
# Phoenix stores OpenInference spans. Export them to a file, then let the `phoenix` adapter
# normalize them into the OTel-JSONL the rest of the pipeline consumes.
set -euo pipefail

# 1) Export spans from Phoenix to a JSON file. Phoenix has no stable file-export CLI, so dump the
#    spans dataframe with the Phoenix client (run against your Phoenix instance):
#
#      python - <<'PY'
#      import phoenix as px
#      df = px.Client().get_spans_dataframe()           # all spans (optionally filter/limit)
#      df.reset_index().to_json("phoenix_export.json", orient="records")
#      PY
#
#    The Phoenix UI's per-trace "Export" also yields a JSON array of span objects. Either shape
#    works: flat OpenInference span dicts (context.trace_id / start_time / attributes) OR an OTLP
#    `resourceSpans` envelope.
EXPORT="${1:-phoenix_export.json}"
MODEL="${2:-phoenix-demo}"

# 2) Build directly from the Phoenix export — `--source phoenix` ingests it as part of build.
uv run wmh build --name "$MODEL" --source phoenix --file "${EXPORT}" --no-interactive

echo "Built world model '$MODEL' from $EXPORT."

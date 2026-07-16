# Ingesting Braintrust traces

The `braintrust` adapter turns a [Braintrust](https://www.braintrust.dev) span-row export into the
normalized `Trace` shape the harness builds world models from. Braintrust does **not** emit OTLP
spans — it logs **spans as rows** in an experiment or project log, where a *trace* is the set of rows
that share a `root_span_id` — so this adapter overrides `spans_from_payload` and re-emits each row in
OTel-GenAI vocabulary for the shared normalizer (`wmh/ingest/normalize.py`).

## The shape

`GET /v1/project_logs/{id}/fetch` (or `/v1/experiment/{id}/fetch`, or the SDK) returns one row per
span, roughly:

```json
{
  "span_id": "s1",
  "root_span_id": "r1",
  "span_parents": [],
  "span_attributes": {"name": "agent", "type": "llm"},
  "input": [{"role": "user", "content": "what's the weather in Paris?"}],
  "output": {"role": "assistant",
             "tool_calls": [{"id": "c1",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
  "metadata": {"model": "gpt-4o"},
  "created": "2026-01-01T00:00:01Z",
  "error": null
}
```

Each row's `span_attributes.type` (commonly `llm | tool | function | task | score`) classifies it.
The adapter maps rows so:

- An **llm / task / function / chain** row whose `output` carries OpenAI-style `tool_calls` becomes a
  `chat` **action** span (`gen_ai.tool.name` + `gen_ai.tool.call.arguments`), one per call. Its
  result is the sibling `tool` row.
- A **tool** row (or an unknown-typed, tool-like row with I/O) becomes an `execute_tool` **result**
  span (`gen_ai.tool.message`), also carrying name/args so a standalone tool row still pairs.
- An **llm / task / function** row with no tool call becomes a plain `chat` message action with no
  observation.
- Rows with no usable output and no tool semantics (e.g. `score`/`eval`) are ignored.
- A non-null `error` marks the row's step as an error.

Rows are grouped by `root_span_id` (the trace key; `span_id` is the per-span key) and ordered by the
`created` ISO-8601 timestamp (-> a monotonic ordinal; list index when absent). The first user message
in `input` becomes the step `task` (`gen_ai.prompt`), and the row `metadata` round-trips via
`wmh.trace.metadata`. The Braintrust ids are used as-is as grouping keys (they need not be 32-hex).

## Export from Braintrust

```bash
# Project logs ({"events": [...]} — the adapter accepts this directly):
curl -s -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  "https://api.braintrust.dev/v1/project_logs/$PROJECT_ID/fetch" > braintrust_export.json

# An experiment's spans:
curl -s -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  "https://api.braintrust.dev/v1/experiment/$EXPERIMENT_ID/fetch" > braintrust_export.json
```

Accepted file shapes: a single span row, a JSON array of rows, an API page wrapper
(`{"events": [...]}` or `{"data": [...]}`), or JSONL (one row per line).

## Run

```bash
uv run wmh build --name braintrust-demo --source braintrust --file braintrust_export.json
```

See `examples/ingest/braintrust_to_wmh.sh` for the end-to-end script.

## Caveats

- **No live pull.** This adapter is file-only; `--pull` raises a friendly error. Export to a file
  first. (The Braintrust SDK would be lazy-imported in `_pull_payloads` if/when pull is added.)
- The richest pairing comes from OpenAI-style `tool_calls` on an llm row's `output`. Custom output
  shapes that don't carry `tool_calls` fall back to a plain message step.
- A tool row's `input` is used as the call arguments only when the preceding llm action lacked them
  (the normalizer backfills name/args from the tool span).
- Field names follow the Braintrust fetch export (`span_id`, `root_span_id`, `span_attributes.type`,
  `created`, `error`). If your export differs (e.g. a flattened SDK dump), the `root_span_id`/
  `span_id`/`span_attributes` keys are what drive grouping and classification.

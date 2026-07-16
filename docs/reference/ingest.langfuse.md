# Ingesting Langfuse traces

The `langfuse` adapter turns a [Langfuse](https://langfuse.com) trace export into the normalized
`Trace` shape the harness builds world models from. Langfuse does **not** emit OTLP spans — it models
a *trace* with a flat list of nested *observations* — so this adapter overrides `spans_from_payload`
and re-emits the observations in OTel-GenAI vocabulary for the shared normalizer
(`wmh/ingest/normalize.py`).

## The shape

`GET /api/public/traces/{id}` (or the SDK) returns roughly:

```json
{
  "id": "lf-trace-abc123",
  "name": "weather-agent",
  "input": "what's the weather in Paris?",
  "metadata": {"benchmark": "demo"},
  "observations": [
    {"id": "o1", "type": "GENERATION", "startTime": "2026-01-01T00:00:01Z",
     "output": {"tool_calls": [{"id": "c1",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}},
    {"id": "o2", "type": "TOOL", "name": "get_weather", "startTime": "2026-01-01T00:00:02Z",
     "input": {"city": "Paris"}, "output": "18C and sunny", "level": "DEFAULT"}
  ]
}
```

Each observation has a `type` of `SPAN | GENERATION | EVENT | TOOL`. The adapter maps them so:

- A **GENERATION** whose `output` carries OpenAI-style `tool_calls` becomes a `chat` **action** span
  (`gen_ai.tool.name` + `gen_ai.tool.call.arguments`). Its result is the sibling tool observation.
- A **TOOL** (or a tool-like **SPAN** with `output`/`input`) becomes an `execute_tool` **result**
  span (`gen_ai.tool.message`), also carrying name/args so a standalone tool observation still pairs.
- A **GENERATION** with no tool call becomes a plain `chat` message action with no observation.
- **EVENT** (and non-actionable) observations are ignored.
- `level == "ERROR"` marks the observation's step as an error (`ObservationLevel` is
  DEBUG | DEFAULT | WARNING | ERROR). `statusMessage` is NOT an error signal — Langfuse sets it on
  any level — so its presence alone does not flag an error.

Observations are ordered by `startTime` (ISO-8601 -> a monotonic ordinal; list index when absent).
The trace `input` becomes the step `task` (`gen_ai.prompt`), and trace `metadata` round-trips via
`wmh.trace.metadata`. The Langfuse trace id is used as-is as the grouping key (it need not be 32-hex).

## Export from Langfuse

```bash
# A single trace by id:
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "$LANGFUSE_HOST/api/public/traces/$TRACE_ID" > langfuse_export.json

# A page of recent traces ({"data": [...]} — the adapter accepts this directly). NOTE: the LIST
# endpoint returns each trace's `observations` as ID *strings* only, so a list page yields no steps —
# fetch each trace by id (loop the ids from this page) to get full observation objects:
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "$LANGFUSE_HOST/api/public/traces?limit=50" > langfuse_traces_page.json
```

Accepted file shapes: a single trace object, a JSON array of traces, an API list page
(`{"data": [...]}`), or JSONL (one trace per line). For a list page to produce steps, each element
must carry full `observations` objects (fetch traces by id), not observation-id strings. For
framework traces where tool calls are separate child observations, Langfuse's native OTLP endpoint
(`POST /api/public/otel/v1/traces`) with `--source otel-genai` is often the cleaner route.

## Run

```bash
uv run wmh build --name langfuse-demo --source langfuse --file langfuse_export.json
```

See `examples/ingest/langfuse_to_wmh.sh` for the end-to-end script.

## Caveats

- **No live pull.** This adapter is file-only; `--pull` raises a friendly error. Export to a file
  first. (The Langfuse SDK would be lazy-imported in `_pull_payloads` if/when pull is added.)
- The richest pairing comes from the OpenAI-style `tool_calls` on a GENERATION `output`. Custom
  output shapes that don't carry `tool_calls` fall back to a plain message step.
- A tool observation's `input` is used as the call arguments only when the preceding GENERATION
  action lacked them (the normalizer backfills name/args from the tool span).

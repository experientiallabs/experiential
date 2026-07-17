# Ingesting traces from anywhere

The harness builds a world model from **recorded agent traces**. Ingestion is part of `wmh build`,
not a separate step: you pick a **source** and `build` turns traces from whatever you already have -
an observability provider, an OTLP export, or a plain chat/tool-call log - into the normalized
`wmh.core.types.Trace` shape and runs the pipeline.

Everything plugs into **one interface** (`TraceAdapter`) and **one normalizer**
(`wmh.ingest.normalize`), so adding a source is a thin adapter, never a rewrite.

## Quickstart

```bash
wmh build --name m --source <name> --file <export>          # build from a file export
wmh build --name m --source <name> --pull --project <p>     # build from a live vendor pull
wmh build                                                   # or pick the source in the wizard
```

`--source` is a registered adapter (`otel-genai`, `chat-json`, `braintrust`, `phoenix`, `langfuse`,
`langsmith`, `posthog`, `mastra`); `--file` reads an export, `--pull` fetches live (with
`--project`/`--api-key`) for
sources that support it. On an interactive terminal, `wmh build` with no source launches a wizard
that lists the sources and prompts for file-or-pull.

Under the hood the chosen adapter normalizes to OTel-GenAI span JSONL - the same format the bundled
`examples/*.otel.jsonl` use - so a source is interchangeable with any other corpus the harness reads.

## Sources

| `--source` | What it reads | File | Live pull |
|---|---|---|---|
| `otel-genai` | OTLP-JSON spans (OTel GenAI semconv) | ✅ | ✅ (generic OTLP query backend) |
| `chat-json` | recorded OpenAI/LangChain-style chat + tool-call conversations | ✅ | - |
| `phoenix` | Arize Phoenix / OpenInference spans | ✅ | provider-dependent |
| `langfuse` | Langfuse trace + observation tree | ✅ | provider-dependent |
| `langsmith` | LangSmith run tree | ✅ | provider-dependent |
| `braintrust` | Braintrust span/log rows | ✅ | ✅ (REST fetch API) |
| `posthog` | PostHog LLM-observability `$ai_*` events | ✅ | ✅ (HogQL query API) |
| `mastra` | Mastra AI-tracing spans (`type` = model_generation/tool_call/…) | ✅ | ✅ (server API) |

Per-provider export instructions, payload shapes, and caveats:
[Phoenix](#ingesting-arize-phoenix-traces) · [Langfuse](#ingesting-langfuse-traces) ·
[LangSmith](#ingesting-langsmith-runs) · [Braintrust](#ingesting-braintrust-traces) ·
[PostHog](#ingesting-posthog-llm-traces) · [Mastra](#ingesting-mastra-traces). Runnable examples
live in [`examples/ingest/`](../../examples/ingest/).

### Already OTel-native? Use `otel-genai` (no dedicated adapter needed)

A dedicated adapter only exists where a provider emits its OWN non-OTLP shape (Langfuse's
observation tree, LangSmith's run tree, Braintrust/PostHog rows, Phoenix's OpenInference dataframe,
Mastra's AI-tracing spans). Platforms that already export **OTel GenAI spans** need no adapter - point
them at a file or an OTLP query backend and use `--source otel-genai`:

- **Traceloop / OpenLLMetry**, **Opik** (Comet), **Logfire** (Pydantic), **Laminar**, **Datadog LLM
  Observability**, **MLflow Tracing**, **Honeycomb**, **New Relic**, **Grafana/Tempo**, **SigNoz** -
  all speak OTLP GenAI natively.
- **Helicone** (LLM gateway/proxy) can forward its logged traffic as OTLP; use `otel-genai` against
  that export. If you only have Helicone's native request/response logs (not OTLP), that's a small
  dedicated adapter - open an issue.
- **Langfuse** and **Mastra** ALSO expose OTLP endpoints; for framework traces where tool calls are
  separate child spans, `otel-genai` against their OTLP export is often cleaner than the native
  adapter (which reads the provider's own tree/span shape).

If a provider isn't listed and doesn't emit OTLP, it's a ~30-line adapter (see below) - open an issue
or add one.

### Databases and other stores

There is no bespoke database adapter - instead, **point your store at OpenTelemetry** and ingest with
`--source otel-genai`. If your agent runs are in a SQL table, a warehouse, or a custom log, the clean
hook-up is to emit OTel GenAI spans (most agent frameworks and the OTel GenAI instrumentations do
this out of the box) and either export them to a file (`--file spans.otlp.json`) or serve them from
an OTLP-compatible query backend and `--pull`. One OTel format, any store behind it - no per-database
code to maintain.

**File vs. pull.** Every adapter supports `--file` (an export you already have); `--pull` (live from
the vendor API) is opt-in per adapter and errors with a clear "export to a file" message when a
source hasn't implemented it. File ingestion needs **no** vendor SDK - the provider adapters parse
the export as JSON. The optional `pip install 'world-model-harness[phoenix]'` (etc.) extras install a
provider's own SDK only if you want to drive its export tooling yourself; nothing in `wmh` imports
them.

## The chat / tool-call converter (`chat-json`)

If you don't use an observability vendor, the most universal trace is a list of chat messages with
tool calls - the OpenAI Chat Completions shape (which LangChain, the Anthropic SDK, and most agent
frameworks can emit). Drop it in a file and ingest it:

```json
{"messages": [
  {"role": "user", "content": "what's the weather in Paris?"},
  {"role": "assistant", "tool_calls": [{"id": "c1",
     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
  {"role": "tool", "tool_call_id": "c1", "content": "18C and sunny"},
  {"role": "assistant", "content": "It's 18C and sunny in Paris."}
]}
```

```bash
wmh build --name my-model --source chat-json --file conversation.json
```

Each assistant tool call becomes an Action paired with its `role:"tool"` result (the Observation);
a trailing assistant message becomes a final message step. Accepts one conversation object, a JSON
array of them, JSONL (one per line), or a bare message list. See `wmh/ingest/messages.py`.

## The trace contract (what an adapter produces)

A `Trace` is `{trace_id, steps, source, metadata}`. Each `Step` is one
`(state_before, action) → observation`:

- `action` - a tool call (`name` + `arguments`) or a free-text message.
- `observation` - what the environment returned (`content`, `is_error`).
- `state_before` - optional env-state snapshot (most provider traces leave it empty; open-loop
  replay reconstructs state from action + history).
- `task` - the originating instruction.

`metadata` carries provenance and anything a source wants to thread through. Open-loop replay scores
a predicted observation for `(state_before, action)` against the recorded one, so faithful `action`
and `observation` are what matter most.

## How the pieces fit

```
                                  ┌─ from_file(path)  ─┐
  raw export / vendor API ──▶ adapter                  ├─▶ list[SpanRecord] ──▶ spans_to_traces ──▶ list[Trace]
                                  └─ from_vendor(pull) ─┘        (wmh.ingest.normalize: the ONE normalizer)
```

- `wmh/ingest/adapter.py` - the `TraceAdapter` protocol + the registry (`register_adapter`,
  `get_adapter`, `list_adapters`).
- `wmh/ingest/base.py` - `BaseTraceAdapter`: file/JSONL loading + vendor plumbing, so a concrete
  adapter only implements `spans_from_payload` (and optionally `_pull_payloads`).
- `wmh/ingest/normalize.py` - the shared span→Trace core. Understands **both** the OTel GenAI
  (`gen_ai.*`) and **OpenInference** (`openinference.span.kind`, `tool.name`, `input.value` /
  `output.value`, `llm.*`) vocabularies, pairs each action span with its following tool span, and
  honors optional `wmh.*` enrichments.
- `wmh/ingest/otel_writer.py` - the inverse: `Trace` → OTel-GenAI span JSONL (used to persist a
  corpus; round-trips losslessly through `otel-genai`).

## Add a new source in ~30 lines

Most providers export spans that are already OTLP or OpenInference, so a new adapter is small:

```python
# wmh/ingest/myprovider.py
from __future__ import annotations

from wmh.ingest.adapter import register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.normalize import SpanRecord


class MyProviderAdapter(BaseTraceAdapter):
    name = "myprovider"

    def spans_from_payload(self, payload):  # one decoded JSON payload -> SpanRecords
        # If the export is OTLP/OpenInference JSON, the BaseTraceAdapter default already works -
        # you don't even need this method. Override it only when the export is a custom shape:
        spans: list[SpanRecord] = []
        for row in payload.get("events", []):
            spans.append(SpanRecord(
                trace_id=row["trace"], span_id=row["id"], start_nano=row["ts"],
                attributes={                      # emit in GenAI vocab; the normalizer pairs them
                    "gen_ai.operation.name": "chat",
                    "gen_ai.tool.name": row["tool"],
                    "gen_ai.tool.call.arguments": row["args"],   # JSON string or object
                },
            ))
            spans.append(SpanRecord(
                trace_id=row["trace"], span_id=row["id"] + "-r", start_nano=row["ts"] + 1,
                status_error=bool(row.get("error")),
                attributes={"gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.message": row["result"]},
            ))
        return spans


register_adapter(MyProviderAdapter())
```

Then import it in `wmh/ingest/__init__.py` (for registration on package import), add an inline
`myprovider_test.py` with a recorded fixture payload (no network), and `wmh build --source myprovider`
picks it up. To support `--pull`, implement `_pull_payloads(pull)` returning raw payloads from the
vendor API (use `httpx`; lazy-import the vendor SDK only if needed). To surface it in the build
wizard's source picker, add it to `_SOURCES` in `wmh/cli/ui.py`. Mirror the four bundled provider
adapters for reference.

## Conventions

Adapters live in `wmh/ingest/`, are typed (no `Any`/bare `dict`; use `wmh.core.types`
`JsonValue`/`JsonObject`), and are tested inline with fixtures - never the network. Vendor SDKs are
optional extras, imported lazily; file ingestion works with none installed.

## Ingesting Arize Phoenix traces

[Arize Phoenix](https://github.com/Arize-ai/phoenix) stores **OpenInference** spans. The `phoenix`
adapter normalizes a Phoenix span export into the harness's OTel-JSONL, reusing the shared
OpenInference classifier (`openinference.span.kind`, `tool.name`, `input.value`, `output.value`,
`llm.input_messages`, `llm.model_name`).

### Export from Phoenix

Phoenix has no stable file-export CLI, so dump spans with the Phoenix client against your instance:

```python
import phoenix as px

df = px.Client().get_spans_dataframe()          # optionally filter / limit
df.reset_index().to_json("phoenix_export.json", orient="records")
```

The Phoenix UI's per-trace **Export** also produces a JSON array of span objects. Both work.

### Shapes the adapter accepts

1. **Phoenix native span dicts** - flat objects where ids live under `context` and timestamps are
   ISO strings:

   ```json
   {
     "name": "agent_step",
     "context": {"trace_id": "f1e2...", "span_id": "aaaa...0001"},
     "parent_id": null,
     "start_time": "2024-01-01T00:00:00.000000+00:00",
     "status_code": "OK",
     "attributes": {"openinference.span.kind": "LLM", "tool.name": "get_user", "...": "..."}
   }
   ```

   A single object, a JSON array, or one object per line (JSONL) are all accepted.

2. **OTLP envelope** - standard `{"resourceSpans": [...]}` OTLP-JSON (or bare OTLP spans with
   `traceId`). These delegate to the shared OTLP collector.

The adapter maps Phoenix's field names (`context.trace_id`, `context.span_id`, `parent_id`,
`start_time`, `status_code`) into the normalizer's span records; ISO timestamps are parsed to epoch
nanoseconds for ordering, falling back to array index when missing/unparseable (ordering only needs
to be monotonic within a trace). LLM/AGENT spans become Actions and the following TOOL span becomes
the Observation, mirroring an agent's `(action) -> observation` step.

### Run it

```bash
uv run wmh build --name phoenix-demo --source phoenix --file phoenix_export.json
```

See `examples/ingest/phoenix_to_wmh.sh` for a runnable script.

### Caveats

- **Live pull is not implemented.** Phoenix's query SDK surface is version-dependent, so the adapter
  leaves `--pull` as the friendly "export to a file" error rather than guess an endpoint. Export to
  a file and use `--file`.
- The adapter is **SDK-free**: it parses the export as JSON and needs no `phoenix`/`arize` package
  installed.

## Ingesting Langfuse traces

The `langfuse` adapter turns a [Langfuse](https://langfuse.com) trace export into the normalized
`Trace` shape the harness builds world models from. Langfuse does **not** emit OTLP spans - it models
a *trace* with a flat list of nested *observations* - so this adapter overrides `spans_from_payload`
and re-emits the observations in OTel-GenAI vocabulary for the shared normalizer
(`wmh/ingest/normalize.py`).

### The shape

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
  DEBUG | DEFAULT | WARNING | ERROR). `statusMessage` is NOT an error signal - Langfuse sets it on
  any level - so its presence alone does not flag an error.

Observations are ordered by `startTime` (ISO-8601 -> a monotonic ordinal; list index when absent).
The trace `input` becomes the step `task` (`gen_ai.prompt`), and trace `metadata` round-trips via
`wmh.trace.metadata`. The Langfuse trace id is used as-is as the grouping key (it need not be 32-hex).

### Export from Langfuse

```bash
# A single trace by id:
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "$LANGFUSE_HOST/api/public/traces/$TRACE_ID" > langfuse_export.json

# A page of recent traces ({"data": [...]} - the adapter accepts this directly). NOTE: the LIST
# endpoint returns each trace's `observations` as ID *strings* only, so a list page yields no steps -
# fetch each trace by id (loop the ids from this page) to get full observation objects:
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "$LANGFUSE_HOST/api/public/traces?limit=50" > langfuse_traces_page.json
```

Accepted file shapes: a single trace object, a JSON array of traces, an API list page
(`{"data": [...]}`), or JSONL (one trace per line). For a list page to produce steps, each element
must carry full `observations` objects (fetch traces by id), not observation-id strings. For
framework traces where tool calls are separate child observations, Langfuse's native OTLP endpoint
(`POST /api/public/otel/v1/traces`) with `--source otel-genai` is often the cleaner route.

### Run

```bash
uv run wmh build --name langfuse-demo --source langfuse --file langfuse_export.json
```

See `examples/ingest/langfuse_to_wmh.sh` for the end-to-end script.

### Caveats

- **No live pull.** This adapter is file-only; `--pull` raises a friendly error. Export to a file
  first. (The Langfuse SDK would be lazy-imported in `_pull_payloads` if/when pull is added.)
- The richest pairing comes from the OpenAI-style `tool_calls` on a GENERATION `output`. Custom
  output shapes that don't carry `tool_calls` fall back to a plain message step.
- A tool observation's `input` is used as the call arguments only when the preceding GENERATION
  action lacked them (the normalizer backfills name/args from the tool span).

## Ingesting LangSmith runs

The `langsmith` adapter turns a [LangSmith](https://smith.langchain.com) (LangChain) run export into
the normalized `Trace` shape the harness builds world models from. LangSmith does **not** emit OTLP
spans - it models a trace as a tree of *runs* - so this adapter overrides `spans_from_payload` and
re-emits the runs in OTel-GenAI vocabulary for the shared normalizer (`wmh/ingest/normalize.py`).

### The shape

`Client.list_runs` (or `POST /api/v1/runs/query` - the list endpoint is a POST with a JSON filter
body, not a GET) returns runs that look roughly like:

```json
[
  {"id": "11...", "trace_id": "tt...", "parent_run_id": null, "run_type": "chain",
   "name": "AgentExecutor", "inputs": {"input": "what's the weather in Paris?"},
   "outputs": {"output": "It's 18C and sunny in Paris."},
   "start_time": "2026-01-01T00:00:00", "error": null},
  {"id": "22...", "trace_id": "tt...", "run_type": "llm", "name": "ChatOpenAI",
   "outputs": {"generations": [{"message": {"kwargs": {"tool_calls": [
       {"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}]}}}]},
   "start_time": "2026-01-01T00:00:01", "error": null},
  {"id": "33...", "trace_id": "tt...", "run_type": "tool", "name": "get_weather",
   "outputs": {"output": "18C and sunny"}, "start_time": "2026-01-01T00:00:03", "error": null}
]
```

Each run is typed by `run_type`. The adapter maps them so:

- An **`llm`** run whose `outputs` carry tool calls becomes one `chat` **action** span per call
  (`gen_ai.tool.name` + `gen_ai.tool.call.arguments`). Its result is the sibling `tool` run.
- An **`llm`** run with no tool call becomes a plain `chat` **message** action (`gen_ai.completion`),
  with no observation.
- A **`tool`** run becomes an `execute_tool` **result** span (`gen_ai.tool.message`).
- **`chain` / `retriever`** (and unknown) runs are not directly actionable and are skipped.
- A non-null `error` marks that run's step as an error.

Tool calls are dug out of the common (version-dependent) LangChain locations, best-effort:
`outputs.generations[].message.kwargs.tool_calls`, the same under `additional_kwargs.tool_calls`
(OpenAI shape), nested list-of-lists generations, and flatter `outputs.tool_calls` /
`outputs.message.kwargs.tool_calls`. A tool-call entry is either the LangChain-normalized
`{"name", "args": {...}}` or the OpenAI `{"function": {"name", "arguments": "<json str>"}}` shape;
both are handled. A run that can't be interpreted is skipped rather than crashing the ingest.

Runs are ordered by `start_time` (ISO-8601 -> a monotonic ordinal; list index when absent), grouped
by `trace_id` (falling back to the run `id` for a root run that omits it). The first human/user
input (dug from a run's `inputs`) becomes the step `task` (`gen_ai.prompt`).

### Export from LangSmith

```bash
# REST API - runs in a project (filter to one trace by id):
curl -s -X POST "${LANGCHAIN_ENDPOINT:-https://api.smith.langchain.com}/api/v1/runs/query" \
  -H "x-api-key: $LANGCHAIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"session": ["<project-uuid>"], "limit": 100}' \
  | python -c 'import sys,json; print(json.dumps(json.load(sys.stdin)["runs"]))' > langsmith_export.json

# SDK (Python):
uv run python -c '
import json, sys
from langsmith import Client
runs = Client().list_runs(project_name="my-project")
json.dump([r.dict() for r in runs], sys.stdout, default=str)' > langsmith_export.json
```

Accepted file shapes: a single run object, a JSON array of runs, a `{"runs": [...]}` wrapper, or
JSONL (one run per line).

### Run

```bash
uv run wmh build --name langsmith-demo --source langsmith --file langsmith_export.json
```

See `examples/ingest/langsmith_to_wmh.sh` for the end-to-end script.

### Caveats

- **No live pull.** This adapter is file-only; `--pull` raises a friendly error. Export to a file
  first. (The `langsmith` SDK would be lazy-imported in `_pull_payloads` if/when pull is added, so
  the config gate stays SDK-free.)
- **Tool-call paths are version-dependent.** LangChain's run-output shape varies across versions;
  the adapter digs the common locations but a custom/unusual dump that doesn't carry `tool_calls`
  falls back to a plain message step.
- Pairing assumes the `tool` run follows its issuing `llm` run by `start_time` (the normalizer pairs
  each action with the nearest following `execute_tool` span). Deeply parallel tool calls within one
  LLM turn pair in start-time order.

## Ingesting Braintrust traces

The `braintrust` adapter turns a [Braintrust](https://www.braintrust.dev) span-row export into the
normalized `Trace` shape the harness builds world models from. Braintrust does **not** emit OTLP
spans - it logs **spans as rows** in an experiment or project log, where a *trace* is the set of rows
that share a `root_span_id` - so this adapter overrides `spans_from_payload` and re-emits each row in
OTel-GenAI vocabulary for the shared normalizer (`wmh/ingest/normalize.py`).

### The shape

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

### Export from Braintrust

```bash
# Project logs ({"events": [...]} - the adapter accepts this directly):
curl -s -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  "https://api.braintrust.dev/v1/project_logs/$PROJECT_ID/fetch" > braintrust_export.json

# An experiment's spans:
curl -s -H "Authorization: Bearer $BRAINTRUST_API_KEY" \
  "https://api.braintrust.dev/v1/experiment/$EXPERIMENT_ID/fetch" > braintrust_export.json
```

Accepted file shapes: a single span row, a JSON array of rows, an API page wrapper
(`{"events": [...]}` or `{"data": [...]}`), or JSONL (one row per line).

### Run

```bash
uv run wmh build --name braintrust-demo --source braintrust --file braintrust_export.json
```

See `examples/ingest/braintrust_to_wmh.sh` for the end-to-end script.

### Caveats

- **No live pull.** This adapter is file-only; `--pull` raises a friendly error. Export to a file
  first. (The Braintrust SDK would be lazy-imported in `_pull_payloads` if/when pull is added.)
- The richest pairing comes from OpenAI-style `tool_calls` on an llm row's `output`. Custom output
  shapes that don't carry `tool_calls` fall back to a plain message step.
- A tool row's `input` is used as the call arguments only when the preceding llm action lacked them
  (the normalizer backfills name/args from the tool span).
- Field names follow the Braintrust fetch export (`span_id`, `root_span_id`, `span_attributes.type`,
  `created`, `error`). If your export differs (e.g. a flattened SDK dump), the `root_span_id`/
  `span_id`/`span_attributes` keys are what drive grouping and classification.

## Ingesting PostHog LLM traces

The `posthog` adapter turns [PostHog LLM observability](https://posthog.com/docs/ai-engineering) data
into the normalized `Trace` shape the harness builds world models from. PostHog captures LLM traces
as analytics **events** (not OTLP spans), so this adapter maps the `$ai_*` events into the OTel-GenAI
vocabulary for the shared normalizer (`wmh/ingest/normalize.py`).

### The shape

Per agent run, PostHog emits events sharing `properties.$ai_trace_id`:

- **`$ai_generation`** - one LLM call. Prompt in `properties.$ai_input` (a messages list), completion
  in `properties.$ai_output_choices`. In PostHog's NORMALIZED shape a tool call is a `content` part
  (`{"type": "function", "function": {"name", "arguments"}}`, with `arguments` a JSON object) and
  assistant text is a `[{"type": "text", "text"}]` parts list; the adapter also accepts a raw-OpenAI
  top-level `tool_calls` array (string `arguments`) for setups that forward the provider payload
  unnormalized.
- **`$ai_span`** - a non-LLM step (often a tool execution): `properties.$ai_span_name` (tool name),
  `properties.$ai_input_state` (args), `properties.$ai_output_state` (result).
- **`$ai_trace`** - a trace-root summary; no standalone step.
- **`$ai_is_error`** (bool) on any event marks that step errored.

The adapter maps each `$ai_generation` tool call to an Action, pairs it with the sibling `$ai_span`
result, and turns a plain generation into a message step. Events order by `timestamp`.

### Build from it

```bash
# From a live PostHog project (HogQL query over $ai_* events):
uv run wmh build --name posthog-demo --source posthog --pull \
  --project "$POSTHOG_PROJECT_ID" --api-key "$POSTHOG_API_KEY"

# Or from an exported events file (a single event, a JSON array, JSONL, or a {"results": [...]}
# HogQL query result):
uv run wmh build --name posthog-demo --source posthog --file events.json
```

- **API key**: a PostHog *personal API key* (Settings → Personal API keys), passed via `--api-key`
  or `$POSTHOG_API_KEY`.
- **Host**: set `$POSTHOG_HOST` for your region (`https://us.posthog.com` default, or
  `https://eu.posthog.com` / a self-hosted URL).
- **Project**: the numeric PostHog project id.

The pull runs `select event, properties, timestamp from events where event like '$ai_%'` via the
HogQL query API and normalizes the rows.

See `examples/ingest/posthog_to_wmh.sh` for a runnable script.

## Ingesting Mastra traces

The `mastra` adapter turns a [Mastra](https://mastra.ai) AI-tracing export into the normalized
`Trace` shape the harness builds world models from. Mastra (a TypeScript agent framework) records
agent runs as **AI-tracing spans** (`ExportedSpan`) typed by `type`, which this adapter maps into the
OTel-GenAI vocabulary for the shared normalizer (`wmh/ingest/normalize.py`). The span id field is
`id`, and spans order by `startTime`. (Mastra renamed its LLM spans to "model" spans in the 2025-11
release; the adapter still accepts the pre-rename aliases - `spanType`, `llm_generation`, `spanId`,
`startedAt` - so older exports keep working.)

### The shape

Spans sharing a `traceId`, each with a `type`:

- **`model_generation`** (pre-rename: `llm_generation`) - an LLM call. `input` is the messages,
  `output` the completion. A completion that issues a tool call carries it as the AI-SDK `toolCalls`
  (`{toolCallId, toolName, input}` on AI SDK v5; `args` on v4) or the OpenAI `tool_calls`
  (`{id, function:{name, arguments}}`) shape.
- **`tool_call`** / **`mcp_tool_call`** - a tool execution: `name` (tool), `input` (args),
  `output` (result).
- **`agent_run`** / **`workflow_*`** / **`model_chunk`** / **`model_step`** / **`generic`** -
  container/streaming spans; no standalone step. An `agent_run`/`model_generation` `input` supplies
  the trace task.
- An `errorInfo`/`error` (or error status) marks the step errored.

Each `model_generation` tool call becomes an Action, paired with the sibling `tool_call` span's
result; a standalone `tool_call` span becomes a complete step on its own.

### Build from it

```bash
# From a running Mastra server (fetches {base}/api/observability/traces):
uv run wmh build --name mastra-demo --source mastra --pull --project http://localhost:4111

# Or from an exported spans file (a single span, a JSON array, a {"spans"|"traces": [...]} wrapper,
# or JSONL):
uv run wmh build --name mastra-demo --source mastra --file mastra_spans.json
```

- **Server URL**: pass the Mastra server base URL as `--project` (or set `$MASTRA_URL`), e.g.
  `http://localhost:4111`. `--api-key` is sent as a bearer token if your server requires auth.
- **File export**: dump Mastra's stored AI spans (from its storage / the observability API) to JSON.

See `examples/ingest/mastra_to_wmh.sh` for a runnable script.

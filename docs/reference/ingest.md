# Local trace input

`wmo build PROJECT TRACES --source SOURCE` reads one explicit local corpus through one canonical
loader. Each source is declared, never guessed:

| `--source` | Local input |
|---|---|
| `otlp` | OpenTelemetry JSON or JSONL using the supported GenAI span mapping. |
| `otel-genai` | Exported flat GenAI span records, re-encoded into OTLP and read by the same mapping. |
| `posthog` | PostHog LLM-observability export. |
| `braintrust` | Braintrust log rows or an `events`, `rows`, `data`, `results`, or `items` envelope. |
| `langfuse` | Langfuse traces with observations, or bare observations carrying `traceId`. |
| `langsmith` | LangSmith runs, or a `runs` envelope. |
| `mastra` | Mastra spans, or a `spans` envelope. |
| `phoenix` | Phoenix and OpenInference spans, native nested, flat dotted, or OTLP JSON. |
| `chat-json` | OpenAI-style chat conversations, one object, an array, or bare message arrays. |
| `postgres` | A JSON declaration of one Postgres table holding rows in one of the formats above. |

Every file source accepts JSON or JSONL. A malformed JSONL line is never skipped silently: it is
retained as an explicit normalization issue. Every normalized trace keeps the immutable source
identity and the exact source-byte digest, the original source trace and span identifiers in
`wmo.source.trace.id` and `wmo.source.span.id`, declared parent relationships when they are
unambiguous, and declared model evidence. Opaque vendor identifiers map to deterministic
W3C-shaped identifiers, so the canonical identity is stable while the source identity stays
readable. Provider and model identity resolves only when the export declares both; a model named
without a provider stays as `gen_ai.request.model` evidence and is never completed by inference.
Tool results pair with tool calls by explicit call identifier, falling back to tool name and source
order only when the export declares no identifier.

A Python caller reaches the same seam by name instead of importing one loader per vendor:

```python
from wmo.simulation.ingest import CANONICAL_TRACE_SOURCES, load_trace_source

result = load_trace_source("langfuse", Path("export.jsonl"))
```

The source table is explicit, so an undeclared name fails closed rather than being detected.

## Postgres tables

`--source postgres` takes a local JSON declaration instead of an export, so a checked-in
declaration never holds a credential:

```json
{
  "table": "public.agent_traces",
  "payload_format": "chat-json",
  "payload_column": "payload",
  "trace_id_column": "trace_id",
  "order_column": "created_at",
  "row_shape": "document",
  "since": "2026-05-01T00:00:00Z"
}
```

The connection string comes from the declaration's optional `dsn` or from `WMO_POSTGRES_DSN`. The
table and column names accept only plain identifiers, and dynamic names reach SQL only as quoted
identifiers. `row_shape` is `document` when one row holds one whole trace payload, or `message`
when one row holds one chat message; message rows require `chat-json`, a `trace_id_column`, and an
`order_column`, and a row with no declared trace identity becomes an explicit issue instead of a
guessed conversation. Turn order is never invented: when the message rows of one conversation share
one order value, that conversation is retained as an explicit issue rather than assembled in an
arbitrary order. Document rows tied on the order column are broken by trace identity and payload
text, so equal timestamps cannot reorder a corpus between builds. `since` requires `order_column`.
The driver is optional: install it with
`uv sync --extra postgres` or `world-model-optimizer[postgres]`. A Python caller may inject its own
row reader through `load_postgres_source(config, reader=...)` and keep its existing pool.

## Stored evidence

Raw exports remain at the customer path. WMO stores a normalized immutable snapshot and explicit
normalization issues under the selected project. Public build accepts 100 through 1000 valid
normalized traces after validation and source deduplication. The limit applies to normalized traces,
not to the smaller representative task count that semantic deduplication and mining produce.

No generic vendor-adapter registry and no format detection are part of the command. Build does not
pull a remote source, resolve a provider, propose a rubric, run a judge, or call an embedding API.
Representative task selection uses the deterministic local hashing embedder unless a Python caller
supplies another explicit descriptor embedder.

Build makes zero model, provider, or judge paid calls. The CLI preserves anonymous aggregate
PostHog product telemetry, which may send after successful persistence unless the user runs
`wmo config telemetry disable`. Telemetry does not include prompts, traces, paths, model names, or
customer content.

## Authorized PostHog HogQL pull

The CLI deliberately reads only local exports. An authorized Python caller may instead import
`PostHogPullRequest` and `pull_posthog_traces` from `wmo.simulation.ingest`. The request defaults to
a bounded 1,000-row query and may set another positive limit through 10,000. The caller may inject a
deterministic HTTP client; otherwise the function owns one bounded `httpx.Client`. The HTTPS host
and credential are explicit request values or resolved from the focused PostHog environment
settings. The query orders by `timestamp, uuid`, applies the same canonical converter as local
export ingestion, and returns normalized traces plus retained issues. This does not weaken the
`wmo build` local-file boundary and requires separate authorization for the customer PostHog
project.

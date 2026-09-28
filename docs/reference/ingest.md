# Local trace input

`exp build PROJECT --traces PATH --source SOURCE` normalizes a local corpus, commits it to
`<root>/gateway/traffic.db` (default root: `.exp`), then mines scenarios and prepares world-model
grounding from that saved evidence. Each source is declared, never guessed:

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
| `experiential` | Completed native JSON Chat Completions capture exports. |
| `gateway` | Retained native captures in a local SQLite database, with required `--identity ID`. |

Every file source accepts JSON or JSONL. A malformed JSONL line is never skipped silently: it is
retained as an explicit normalization issue. Every normalized trace keeps the immutable source
identity and the exact source-byte digest, the original source trace and span identifiers in
`exp.source.trace.id` and `exp.source.span.id`, declared parent relationships when they are
unambiguous, and declared model evidence. Opaque vendor identifiers map to deterministic
W3C-shaped identifiers, so the canonical identity is stable while the source identity stays
readable. Provider and model identity resolves only when the export declares both; a model named
without a provider stays as `gen_ai.request.model` evidence and is never completed by inference.
Tool results pair with tool calls by explicit call identifier, falling back to tool name and source
order only when the export declares no identifier.

A Python caller selects a file source by name through shared ingestion:

```python
from pathlib import Path
from exp.common.traces.ingest import CANONICAL_TRACE_SOURCES, load_trace_source

result = load_trace_source("langfuse", Path("export.jsonl"))
```

The file-source table is explicit, so an undeclared name fails closed rather than being detected.
File parsing, normalization, provenance and persistence live in `exp/common/traces/ingest/` and
do not import runtime or simulation code. Gateway capture and protocol adapters live in
`exp/runtime/gateway/ingest/`; the CLI selects the adapter for `--source gateway`.

## Stored evidence

The ingestion stage writes canonical traces and provenance into the same SQLite content database as
local gateway capture. Source files stay at the caller's path. The database owns normalized trace
records, immutable imports, ordered import membership and project associations. The same database
also owns project configuration, immutable dataset/scenario and judge versions, evaluation runs,
checkpoints, judgments, and result metadata. The lower-level Python import API does not require an
existing project configuration; the build command also prepares project state and scenarios.

Canonical content is hashed independently of per-import source provenance, so overlapping exports
can share identical normalized records. An import ID binds the source format, source digest,
ordered trace records, normalization exclusions and model identity evidence. Repeat ingestion is
idempotent. Changed evidence creates another immutable import, including changes to a trace that
reuses its original ID. Imported evidence survives native capture expiry and count/byte pruning.
Deleting the shared database also deletes those imports; preserve it when clearing raw captures.

Ingest snapshots and parses source archives incrementally into private temporary disk storage,
groups interleaved spans there, and normalizes one complete trace at a time. Payload memory follows
the largest source record or trace, not the total archive. Compact exclusion and model-identity
metadata is collected separately and still grows with record count. Temporary disk space is needed
for the source snapshot, indexed records and normalized output; it is removed on ordinary completion
or exception. An abruptly killed process can leave temporary files for operating-system cleanup.
These memory bounds apply to source ingestion. Scenario mining and RAG preparation currently
materialize the canonical corpus, so a complete build can use memory proportional to its size.

Canonical records and import membership are written in short transactions targeting at most
64 records or 1 MiB of serialized records per batch. These are buffer targets, not input limits:
a larger individual trace is accepted intact. The final transaction publishes the completed import
through its project association. Unlinked staging is invisible to import readers. An interrupted
write retains verified staging that an identical retry can complete; changed source evidence gets
a different import identity. Published evidence is never repaired silently. Raw SQL inspection can
see staging, so consumers must use the store's read methods.

Gateway collection can acquire the writer lock between batches. Reads use their own SQLite
snapshot. Unsupported database schemas fail before any settings or table changes.

`exp build --dry-run` uses temporary SQLite for ingestion. It may checkpoint deterministic project
artifacts, but writes no shared trace import, makes no provider calls, and selects no completed
build. The lower-level Python import API's `dry_run=True` writes no persistent storage. Ingestion
has no trace-count minimum or total-count ceiling and performs no model calls; build requires at
least one valid trace and uses the configured embedder after its cost preflight.
`--source gateway --identity ID` defaults to this root's traffic database and reads all retained
records for that identity, copying one row at a time from one snapshot to private disk. `--traces PATH` can select
another explicit capture database. Identity is never inferred from the project name.

Python callers can restore complete normalized evidence after the original export disappears:

```python
from pathlib import Path
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.common.traces.ingest.persistence import ingest_traces, read_ingested_traces

root = Path(".exp")
summary, receipt = ingest_traces(
    "powerset", root=root, path=Path("rollouts.jsonl"), source_format="chat-json"
)
assert receipt is not None
store = SQLiteTraceStore(trace_database_path(root))
import_ids = store.list_imports("powerset")
restored = read_ingested_traces(root, receipt.import_id)
assert len(restored.traces) == summary.trace_count
assert restored.issues == summary.issues
assert restored.source == summary.source
```

Python callers ingest gateway captures through the explicitly scoped runtime adapter, which uses
the same shared normalization workspace and SQLite import writer:

```python
from pathlib import Path
from exp.runtime.gateway.ingest import ingest_gateway_capture

summary, receipt = ingest_gateway_capture(
    "powerset", root=Path(".exp"), path=Path(".exp/gateway/traffic.db"), identity_id="default"
)
```

Ingestion is the first stage of `exp build`. The command then mines representative tasks,
selects project model roles, estimates embedding cost, builds serving and fit-only RAG indexes,
and binds the grounded world model under the configured spend ceiling. Exact repeated builds
reuse their saved evidence and completed indexes. Source imports and scenario artifacts remain
immutable, preserving the evidence pinned by earlier evaluations. Preparation alone produces no
evaluation or HTML report; those are outputs of running and judging candidate rollouts.

Select an exact stored corpus with `exp build PROJECT --import-id IMPORT_ID --root ROOT`.
The build pins that import in its configuration and does not reread the original source.

## Authorized PostHog HogQL pull

The CLI deliberately reads only local exports. An authorized Python caller may instead import
`PostHogPullRequest` and `pull_posthog_traces` from `exp.common.traces.ingest`. The request defaults to
a bounded 1,000-row query and may set another positive limit through 10,000. The caller may inject a
deterministic HTTP client; otherwise the function owns one bounded `httpx.Client`. The HTTPS host
and credential are explicit request values or resolved from the focused PostHog environment
settings. The query orders by `timestamp, uuid`, applies the same canonical converter as local
export ingestion, and returns normalized traces plus retained issues. This does not weaken the
`exp build` local-file boundary and requires separate authorization for the customer PostHog
project.

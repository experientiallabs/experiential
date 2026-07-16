# Ingesting traces from anywhere

The harness builds a world model from **recorded agent traces**. Ingestion is part of `wmh build`,
not a separate step: you pick a **source** and `build` turns traces from whatever you already have —
an observability provider, an OTLP export, or a plain chat/tool-call log — into the normalized
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

Under the hood the chosen adapter normalizes to OTel-GenAI span JSONL — the same format the bundled
`examples/*.otel.jsonl` use — so a source is interchangeable with any other corpus the harness reads.

## Sources

| `--source` | What it reads | File | Live pull |
|---|---|---|---|
| `otel-genai` | OTLP-JSON spans (OTel GenAI semconv) | ✅ | ✅ (generic OTLP query backend) |
| `chat-json` | recorded OpenAI/LangChain-style chat + tool-call conversations | ✅ | — |
| `phoenix` | Arize Phoenix / OpenInference spans | ✅ | provider-dependent |
| `langfuse` | Langfuse trace + observation tree | ✅ | provider-dependent |
| `langsmith` | LangSmith run tree | ✅ | provider-dependent |
| `braintrust` | Braintrust span/log rows | ✅ | ✅ (REST fetch API) |
| `posthog` | PostHog LLM-observability `$ai_*` events | ✅ | ✅ (HogQL query API) |
| `mastra` | Mastra AI-tracing spans (`type` = model_generation/tool_call/…) | ✅ | ✅ (server API) |

Per-provider export instructions, payload shapes, and caveats:
[Phoenix](./ingest.phoenix.md) · [Langfuse](./ingest.langfuse.md) ·
[LangSmith](./ingest.langsmith.md) · [Braintrust](./ingest.braintrust.md) ·
[PostHog](./ingest.posthog.md) · [Mastra](./ingest.mastra.md). Runnable examples live in
[`examples/ingest/`](../examples/ingest/).

### Already OTel-native? Use `otel-genai` (no dedicated adapter needed)

A dedicated adapter only exists where a provider emits its OWN non-OTLP shape (Langfuse's
observation tree, LangSmith's run tree, Braintrust/PostHog rows, Phoenix's OpenInference dataframe,
Mastra's AI-tracing spans). Platforms that already export **OTel GenAI spans** need no adapter — point
them at a file or an OTLP query backend and use `--source otel-genai`:

- **Traceloop / OpenLLMetry**, **Opik** (Comet), **Logfire** (Pydantic), **Laminar**, **Datadog LLM
  Observability**, **MLflow Tracing**, **Honeycomb**, **New Relic**, **Grafana/Tempo**, **SigNoz** —
  all speak OTLP GenAI natively.
- **Helicone** (LLM gateway/proxy) can forward its logged traffic as OTLP; use `otel-genai` against
  that export. If you only have Helicone's native request/response logs (not OTLP), that's a small
  dedicated adapter — open an issue.
- **Langfuse** and **Mastra** ALSO expose OTLP endpoints; for framework traces where tool calls are
  separate child spans, `otel-genai` against their OTLP export is often cleaner than the native
  adapter (which reads the provider's own tree/span shape).

If a provider isn't listed and doesn't emit OTLP, it's a ~30-line adapter (see below) — open an issue
or add one.

### Databases and other stores

There is no bespoke database adapter — instead, **point your store at OpenTelemetry** and ingest with
`--source otel-genai`. If your agent runs are in a SQL table, a warehouse, or a custom log, the clean
hook-up is to emit OTel GenAI spans (most agent frameworks and the OTel GenAI instrumentations do
this out of the box) and either export them to a file (`--file spans.otlp.json`) or serve them from
an OTLP-compatible query backend and `--pull`. One OTel format, any store behind it — no per-database
code to maintain.

**File vs. pull.** Every adapter supports `--file` (an export you already have); `--pull` (live from
the vendor API) is opt-in per adapter and errors with a clear "export to a file" message when a
source hasn't implemented it. File ingestion needs **no** vendor SDK — the provider adapters parse
the export as JSON. The optional `pip install 'world-model-harness[phoenix]'` (etc.) extras install a
provider's own SDK only if you want to drive its export tooling yourself; nothing in `wmh` imports
them.

## The chat / tool-call converter (`chat-json`)

If you don't use an observability vendor, the most universal trace is a list of chat messages with
tool calls — the OpenAI Chat Completions shape (which LangChain, the Anthropic SDK, and most agent
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

- `action` — a tool call (`name` + `arguments`) or a free-text message.
- `observation` — what the environment returned (`content`, `is_error`).
- `state_before` — optional env-state snapshot (most provider traces leave it empty; open-loop
  replay reconstructs state from action + history).
- `task` — the originating instruction.

`metadata` carries provenance and anything a source wants to thread through. Open-loop replay scores
a predicted observation for `(state_before, action)` against the recorded one, so faithful `action`
and `observation` are what matter most.

## How the pieces fit

```
                                  ┌─ from_file(path)  ─┐
  raw export / vendor API ──▶ adapter                  ├─▶ list[SpanRecord] ──▶ spans_to_traces ──▶ list[Trace]
                                  └─ from_vendor(pull) ─┘        (wmh.ingest.normalize: the ONE normalizer)
```

- `wmh/ingest/adapter.py` — the `TraceAdapter` protocol + the registry (`register_adapter`,
  `get_adapter`, `list_adapters`).
- `wmh/ingest/base.py` — `BaseTraceAdapter`: file/JSONL loading + vendor plumbing, so a concrete
  adapter only implements `spans_from_payload` (and optionally `_pull_payloads`).
- `wmh/ingest/normalize.py` — the shared span→Trace core. Understands **both** the OTel GenAI
  (`gen_ai.*`) and **OpenInference** (`openinference.span.kind`, `tool.name`, `input.value` /
  `output.value`, `llm.*`) vocabularies, pairs each action span with its following tool span, and
  honors optional `wmh.*` enrichments.
- `wmh/ingest/otel_writer.py` — the inverse: `Trace` → OTel-GenAI span JSONL (used to persist a
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
        # If the export is OTLP/OpenInference JSON, the BaseTraceAdapter default already works —
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
`JsonValue`/`JsonObject`), and are tested inline with fixtures — never the network. Vendor SDKs are
optional extras, imported lazily; file ingestion works with none installed.

# Ingesting Mastra traces

The `mastra` adapter turns a [Mastra](https://mastra.ai) AI-tracing export into the normalized
`Trace` shape the harness builds world models from. Mastra (a TypeScript agent framework) records
agent runs as **AI-tracing spans** (`ExportedSpan`) typed by `type`, which this adapter maps into the
OTel-GenAI vocabulary for the shared normalizer (`wmh/ingest/normalize.py`). The span id field is
`id`, and spans order by `startTime`. (Mastra renamed its LLM spans to "model" spans in the 2025-11
release; the adapter still accepts the pre-rename aliases — `spanType`, `llm_generation`, `spanId`,
`startedAt` — so older exports keep working.)

## The shape

Spans sharing a `traceId`, each with a `type`:

- **`model_generation`** (pre-rename: `llm_generation`) — an LLM call. `input` is the messages,
  `output` the completion. A completion that issues a tool call carries it as the AI-SDK `toolCalls`
  (`{toolCallId, toolName, input}` on AI SDK v5; `args` on v4) or the OpenAI `tool_calls`
  (`{id, function:{name, arguments}}`) shape.
- **`tool_call`** / **`mcp_tool_call`** — a tool execution: `name` (tool), `input` (args),
  `output` (result).
- **`agent_run`** / **`workflow_*`** / **`model_chunk`** / **`model_step`** / **`generic`** —
  container/streaming spans; no standalone step. An `agent_run`/`model_generation` `input` supplies
  the trace task.
- An `errorInfo`/`error` (or error status) marks the step errored.

Each `model_generation` tool call becomes an Action, paired with the sibling `tool_call` span's
result; a standalone `tool_call` span becomes a complete step on its own.

## Build from it

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

# Ingesting LangSmith runs

The `langsmith` adapter turns a [LangSmith](https://smith.langchain.com) (LangChain) run export into
the normalized `Trace` shape the harness builds world models from. LangSmith does **not** emit OTLP
spans — it models a trace as a tree of *runs* — so this adapter overrides `spans_from_payload` and
re-emits the runs in OTel-GenAI vocabulary for the shared normalizer (`wmh/ingest/normalize.py`).

## The shape

`Client.list_runs` (or `POST /api/v1/runs/query` — the list endpoint is a POST with a JSON filter
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

## Export from LangSmith

```bash
# REST API — runs in a project (filter to one trace by id):
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

## Run

```bash
uv run wmh build --name langsmith-demo --source langsmith --file langsmith_export.json
```

See `examples/ingest/langsmith_to_wmh.sh` for the end-to-end script.

## Caveats

- **No live pull.** This adapter is file-only; `--pull` raises a friendly error. Export to a file
  first. (The `langsmith` SDK would be lazy-imported in `_pull_payloads` if/when pull is added, so
  the config gate stays SDK-free.)
- **Tool-call paths are version-dependent.** LangChain's run-output shape varies across versions;
  the adapter digs the common locations but a custom/unusual dump that doesn't carry `tool_calls`
  falls back to a plain message step.
- Pairing assumes the `tool` run follows its issuing `llm` run by `start_time` (the normalizer pairs
  each action with the nearest following `execute_tool` span). Deeply parallel tool calls within one
  LLM turn pair in start-time order.

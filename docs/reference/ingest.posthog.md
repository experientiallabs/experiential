# Ingesting PostHog LLM traces

The `posthog` adapter turns [PostHog LLM observability](https://posthog.com/docs/ai-engineering) data
into the normalized `Trace` shape the harness builds world models from. PostHog captures LLM traces
as analytics **events** (not OTLP spans), so this adapter maps the `$ai_*` events into the OTel-GenAI
vocabulary for the shared normalizer (`wmh/ingest/normalize.py`).

## The shape

Per agent run, PostHog emits events sharing `properties.$ai_trace_id`:

- **`$ai_generation`** — one LLM call. Prompt in `properties.$ai_input` (a messages list), completion
  in `properties.$ai_output_choices`. In PostHog's NORMALIZED shape a tool call is a `content` part
  (`{"type": "function", "function": {"name", "arguments"}}`, with `arguments` a JSON object) and
  assistant text is a `[{"type": "text", "text"}]` parts list; the adapter also accepts a raw-OpenAI
  top-level `tool_calls` array (string `arguments`) for setups that forward the provider payload
  unnormalized.
- **`$ai_span`** — a non-LLM step (often a tool execution): `properties.$ai_span_name` (tool name),
  `properties.$ai_input_state` (args), `properties.$ai_output_state` (result).
- **`$ai_trace`** — a trace-root summary; no standalone step.
- **`$ai_is_error`** (bool) on any event marks that step errored.

The adapter maps each `$ai_generation` tool call to an Action, pairs it with the sibling `$ai_span`
result, and turns a plain generation into a message step. Events order by `timestamp`.

## Build from it

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

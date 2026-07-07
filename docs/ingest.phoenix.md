# Ingesting Arize Phoenix traces

[Arize Phoenix](https://github.com/Arize-ai/phoenix) stores **OpenInference** spans. The `phoenix`
adapter normalizes a Phoenix span export into the harness's OTel-JSONL, reusing the shared
OpenInference classifier (`openinference.span.kind`, `tool.name`, `input.value`, `output.value`,
`llm.input_messages`, `llm.model_name`).

## Export from Phoenix

Phoenix has no stable file-export CLI, so dump spans with the Phoenix client against your instance:

```python
import phoenix as px

df = px.Client().get_spans_dataframe()          # optionally filter / limit
df.reset_index().to_json("phoenix_export.json", orient="records")
```

The Phoenix UI's per-trace **Export** also produces a JSON array of span objects. Both work.

## Shapes the adapter accepts

1. **Phoenix native span dicts** — flat objects where ids live under `context` and timestamps are
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

2. **OTLP envelope** — standard `{"resourceSpans": [...]}` OTLP-JSON (or bare OTLP spans with
   `traceId`). These delegate to the shared OTLP collector.

The adapter maps Phoenix's field names (`context.trace_id`, `context.span_id`, `parent_id`,
`start_time`, `status_code`) into the normalizer's span records; ISO timestamps are parsed to epoch
nanoseconds for ordering, falling back to array index when missing/unparseable (ordering only needs
to be monotonic within a trace). LLM/AGENT spans become Actions and the following TOOL span becomes
the Observation, mirroring an agent's `(action) -> observation` step.

## Run it

```bash
uv run wmh build --name phoenix-demo --source phoenix --file phoenix_export.json
```

See `examples/ingest/phoenix_to_wmh.sh` for a runnable script.

## Caveats

- **Live pull is not implemented.** Phoenix's query SDK surface is version-dependent, so the adapter
  leaves `--pull` as the friendly "export to a file" error rather than guess an endpoint. Export to
  a file and use `--file`.
- The adapter is **SDK-free**: it parses the export as JSON and needs no `phoenix`/`arize` package
  installed.

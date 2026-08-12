# Local trace input

`wmo build` accepts two canonical local inputs:

- `--source otlp` reads OpenTelemetry JSON or JSONL using the supported GenAI span mapping.
- `--source posthog` reads a local PostHog LLM-observability export.

The positional `TRACE_FILE` and `--project` are required. Raw exports remain at the customer path.
WMO stores a normalized immutable snapshot and explicit normalization issues under the selected
project. Public build accepts 100 through 1000 valid normalized traces after validation and source
deduplication. The limit applies to normalized traces, not to the smaller representative task count
that semantic deduplication and mining produce.

No generic vendor-adapter registry is part of the command. Build does not pull a remote source,
resolve a provider, propose a rubric, run a judge, or call an embedding API. Representative task
selection uses the deterministic local hashing embedder unless a Python caller supplies another
explicit descriptor embedder.

Build makes zero model, provider, or judge paid calls. The CLI preserves anonymous aggregate
PostHog product telemetry, which may send after successful persistence unless the user runs
`wmo config telemetry disable`. Telemetry does not include prompts, traces, paths, model names, or
customer content.

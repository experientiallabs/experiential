# Local trace input

`wmo build` accepts two canonical local inputs:

- `--source otlp` reads OpenTelemetry JSON or JSONL using the supported GenAI span mapping.
- `--source posthog` reads a local PostHog LLM-observability export.

The positional `TRACE_FILE` and `--project` are required. Raw exports remain at the customer path.
WMO stores a normalized immutable snapshot and explicit normalization issues under the selected
project.

No generic vendor-adapter registry is part of the command. Build does not pull a remote source,
resolve a provider, propose a rubric, run a judge, or call an embedding API. Representative task
selection uses the deterministic local hashing embedder unless a Python caller supplies another
explicit descriptor embedder.

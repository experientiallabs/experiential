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

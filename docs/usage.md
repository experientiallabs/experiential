# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `wmo build TRACE_FILE --project PROJECT --root ROOT` | Normalize local OTLP or PostHog evidence and mine representative tasks. | `TraceDataset`, `TaskSet`, and `proposals_pending` review state. |
| `wmo optimize router PROJECT --config FILE --root ROOT` | Fit, freeze, then report from explicit completed evidence. | Fit evaluation, bank, policy, held-out evaluation, and router report. |
| `wmo run PROJECT --root ROOT` | Load a frozen policy and expose it on development-only loopback. | Local OpenAI-compatible endpoint requiring `X-WMO-Episode-ID`. |
| `wmo config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.wmo/settings.toml`. |

`build` and `optimize router` do not make provider calls. `run` makes no provider call at startup.
An HTTP completion or direct `RouterRuntime.complete` call is the explicit online model-call
boundary. The router remains frozen for the lifetime of the process.

Removed commands and aliases include `providers`, `list`, `download`, `eval`, `knowledge`,
`serve`, `optimize route`, and the separate router `fit` and `report` commands.

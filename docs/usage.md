# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `wmo build TRACE_FILE --project PROJECT --root ROOT` | Normalize 100 through 1000 local OTLP or PostHog traces and mine representative tasks. | Manifest-bound `TraceDataset`, `TaskSet`, and `proposals_pending` review state. |
| `wmo optimize router PROJECT --config FILE --root ROOT` | Fit, freeze, then report from explicit completed evidence. | Fit evaluation, bank, policy, held-out evaluation, and router report. |
| `wmo optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `wmo run PROJECT --root ROOT [--ghost]` | Load a frozen policy and expose it on development-only loopback. | Local OpenAI-compatible endpoint with durable journaling by default or no saved traffic in ghost mode. |
| `wmo config providers` | Collect secret-free provider connections, model aliases, and build roles. | Local `.wmo/models.toml`. |
| `wmo config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.wmo/settings.toml`. |

`build` makes zero model, provider, or judge paid calls. Successful build, router, simulation, and
SFT operations preserve anonymous aggregate PostHog product telemetry, which may send unless
disabled. `optimize router` does not make provider calls. `optimize model` requires an immutable
positive cost ceiling, a finite conservative full-schedule estimate, explicit consent, and complete
project evidence before Tinker can open. Unknown cost or incomplete provenance fails before
dispatch. `run` makes no provider call at startup.
An HTTP completion or direct `RouterRuntime.complete` call is the explicit online model-call
boundary. The router remains frozen for the process lifetime. `run --ghost` still permits provider
calls but disables durable interaction, replay, RAG, and SFT state for that process.

Build stops at review readiness. Simulation, judgment, fidelity, embedding, and pricing artifacts
must be completed through a separately authorized workflow before creating the exact
[`router-optimization.json`](reference/router_optimization_config.md) input.

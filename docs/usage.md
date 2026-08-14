# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `wmo build PROJECT TRACES --root ROOT [--dry-run]` | Normalize local OTLP or PostHog traces, mine representative tasks, and ground serving plus fit-only retrieval when the embedding estimate is within `--max-build-cost-usd`. | Manifest-bound `TraceDataset`, `TaskSet`, serving RAG, fit-only RAG, grounded world model, and `proposals_pending` review state. |
| `wmo optimize router PROJECT --config FILE --root ROOT` | Fit, freeze, then report from explicit completed evidence. | Fit evaluation, bank, policy, held-out evaluation, and router report. |
| `wmo optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `wmo run PROJECT --root ROOT [--ghost]` | Load a frozen policy and expose it on development-only loopback. | Local OpenAI-compatible endpoint with durable journaling by default or no saved traffic in ghost mode. |
| `wmo config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.wmo/settings.toml`. |

`build` authorizes provider embeddings when the conservative estimate is within
`--max-build-cost-usd`. It makes no judge or completion calls. `--dry-run` prints the complete
preflight without credentials, provider calls, or a completed-build selection. Successful build,
router, simulation, and SFT operations preserve anonymous aggregate PostHog product telemetry,
which may send unless disabled. `optimize router` does not make provider calls. `optimize model`
requires an immutable positive cost ceiling, a finite conservative full-schedule estimate, explicit
consent, and complete project evidence before Tinker can open. Unknown cost or incomplete
provenance fails before dispatch. `run` makes no provider call at startup.
An HTTP completion or direct `RouterRuntime.complete` call is the explicit online model-call
boundary. The router remains frozen for the process lifetime. `run --ghost` still permits provider
calls but disables durable interaction, replay, RAG, and SFT state for that process.

Build stops at review readiness after grounding retrieval. Simulation, judgment, fidelity, and
pricing artifacts must be completed through a separately authorized workflow before creating the
exact [`router-optimization.json`](reference/router_optimization_config.md) input.

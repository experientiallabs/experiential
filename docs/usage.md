# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `wmo build PROJECT TRACES --source SOURCE --root ROOT [--provider NAME ...]` | Normalize 100 through 1000 local traces from one [declared source](reference/ingest.md) and mine representative tasks. First-build setup uses a keyboard provider list, or exact `--provider` flags. | Manifest-bound `TraceDataset`, `TaskSet`, and `proposals_pending` review state. |
| `wmo optimize router PROJECT --root ROOT [--yes]` | Complete bounded fit simulation and judgment, lock a frozen router, then verify held-out evidence. | Fit evaluation, policy, held-out evaluation, and router report. |
| `wmo optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `wmo run PROJECT --root ROOT [--ghost]` | Load a frozen policy and expose it on development-only loopback. | Local OpenAI-compatible endpoint with durable journaling by default or no saved traffic in ghost mode. |
| `wmo config providers [--provider NAME ...]` | Collect secret-free provider connections, model aliases, and build roles. | Local `.wmo/models.toml`. |
| `wmo config budget [USD] --root ROOT` | Read or set the maximum conservative estimate allowed for one paid command. | Local `.wmo/settings.toml`. |
| `wmo config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.wmo/settings.toml`. |

`build`, judge calibration, `optimize router`, and `optimize model` use the same cost authorization
policy. Each displays the command, conservative estimate, configured per-command budget, and major
cost assumptions. An estimate at or below 50% of the budget runs automatically. A higher estimate
up to the budget requires a clear terminal confirmation or `--yes`; an estimate above the budget
fails before credentials or provider clients. Set the deterministic ceiling with
`wmo config budget USD --root ROOT`. `--yes` confirms only an in-budget invocation and never raises
the ceiling. Exact completed replays report a zero-dollar estimate and do not prompt.

Successful build, router, simulation, and SFT operations preserve anonymous aggregate PostHog
product telemetry, which may send unless disabled. `run` makes no provider call at startup.
An HTTP completion or direct `RouterRuntime.complete` call is the explicit online model-call
boundary. The router remains frozen for the process lifetime. `run --ghost` still permits provider
calls but disables durable interaction, replay, RAG, and SFT state for that process.

Build stops at review readiness. `wmo optimize router` creates the bounded fit and held-out
evaluation chain after candidate and manual judge setup. It never invokes world-model fidelity
testing. Applications that need a world-model quality measurement can call the separate
`build_fidelity_evaluation_plan` and `build_fidelity_report` APIs; those results never enter router
fitting or activation. Fidelity reports contain measurements only, never an approval or denial.
See the [router contracts](reference/router_optimization_config.md).

# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `wmo build PROJECT [-t PATH] --source SOURCE --root ROOT [--provider NAME ...]` | Launch the guided end-to-end build when traces are omitted, or normalize the explicit local source for automation. | Manifest-bound trace and task evidence, serving and fit-only RAG, grounded world model, judge review state, and automatic router artifacts. |
| `wmo optimize router PROJECT --root ROOT [--yes]` | Complete bounded simulation and judgment, fit a frozen router, then verify held-out evidence. | Fit evaluation, policy, held-out evaluation, and router report. |
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
`build --dry-run` and exact completed-build replay make zero paid provider calls. A new grounded
build calls only the configured embedder; automatic router optimization separately executes the
bounded candidate, world-model, and judge schedule shown in its cost preflight.
An HTTP completion or direct `RouterRuntime.complete` call is the explicit online model-call
boundary. The router remains frozen for the process lifetime. `run --ghost` still permits provider
calls but disables durable interaction, replay, RAG, and SFT state for that process.

Build stops at review readiness. Simulation, judgment, fidelity, embedding, and pricing artifacts
must be completed through a separately authorized workflow before creating the exact
[`router-optimization.json`](reference/router_optimization_config.md) input.

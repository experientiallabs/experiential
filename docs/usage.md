# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `wmo build PROJECT TRACES --source SOURCE --root ROOT [--provider NAME ...]` | Normalize 100 through 1000 local traces from one [declared source](reference/ingest.md) and mine representative tasks. First-build setup uses a keyboard provider list, or exact `--provider` flags. | Manifest-bound `TraceDataset`, `TaskSet`, and `proposals_pending` review state. |
| `wmo optimize router PROJECT --root ROOT [--yes]` | Complete bounded fit simulation and judgment, lock a frozen router, then verify held-out evidence. | Fit evaluation, policy, held-out evaluation, and router report. |
| `wmo optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `wmo run --root ROOT [--check]` | Validate or start the initialized authenticated multi-alias gateway on loopback. | OpenAI-compatible endpoint, readiness routes, and content-free usage view. |
| `wmo run PROJECT --root ROOT [--ghost]` | Load a frozen policy and expose it on development-only loopback. | Local OpenAI-compatible endpoint with durable journaling by default or no saved traffic in ghost mode. |
| `wmo config gateway ...` | Author provider references, identities, virtual keys, grants, aliases, certified exact-model pools, status, and usage without optimizer roles. | Private SQLite authority, immutable catalog snapshots, and versioned receipts. |
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

No-argument `wmo run` is a separate gateway lifecycle. It binds only `127.0.0.1`, starts with no
provider call, and requires an explicit provider environment reference, exact model alias, identity,
grant, and virtual key. `wmo run --non-interactive --json` returns `gateway_not_initialized` plus
exact next commands on an empty root. `wmo run --check` validates local readiness without binding.
The gateway writes no prompts, responses, tool arguments, raw keys, or provider secrets to SQLite.
`GET /usage` and `GET /usage.json` show only per-identity counts, token usage, latency, terminal
states, and attributed estimated cost. Estimated cost is not provider invoice cost.

One-time virtual-key material appears only in the successful key-issue receipt or a newly created
mode-`0600` output file. Human key issuance on a non-terminal requires `--json` or `--output`.
Provider configuration accepts an environment variable name, never a raw credential value.

To add ordered failover, first author each deployment as a direct alias with the same
`--exact-model`. Then certify their equivalence and order with:

```console
wmo config gateway pool certify PUBLIC_ALIAS \
  --deployment-alias PRIMARY --deployment-alias SECONDARY \
  --exact-model EXACT_MODEL --certification-id CERTIFICATION \
  --provenance PROVENANCE --evidence-sha256 SHA256 \
  --certified-at TIMESTAMP --expected-catalog-sha256 CATALOG_SHA256 \
  --revision REVISION --root ROOT --non-interactive --json
```

`--expected-catalog-sha256` prevents stale authoring from activating a different catalog. Every
pool member must resolve to the same exact model identity. Retryable transport, availability, rate,
and malformed precommit failures may advance through the certified order. Refusal fallback requires
the explicit `--refusal-failover` option and is persisted on that alias revision. No failure can
switch providers after outward text, refusal, or tool-call output commits the response.

Official OpenAI SDK clients use the issued virtual key and loopback base URL:

```python
from openai import OpenAI

with OpenAI(api_key=VIRTUAL_KEY, base_url="http://127.0.0.1:8000/v1") as client:
    response = client.responses.create(model="PUBLIC_ALIAS", input="hello")
```

Release evidence fixes the SDK at OpenAI `3.0.0` and covers `OpenAI` plus `AsyncOpenAI`, Chat
Completions plus Responses, and stream plus non-stream calls. Provider protocol fixtures are
deterministic. Hosted-provider runs require credentials and are reported separately in
[`release-scope.md`](release-scope.md); fixture success is not presented as a live-provider result.

Build stops at review readiness. `wmo optimize router` creates the bounded fit and held-out
evaluation chain after candidate and manual judge setup. It never invokes world-model fidelity
testing. Applications that need a world-model quality measurement can call the separate
`build_fidelity_evaluation_plan` and `build_fidelity_report` APIs; those results never enter router
fitting or activation. Fidelity reports contain measurements only, never an approval or denial.
See the [router contracts](reference/router_optimization_config.md).

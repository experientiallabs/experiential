# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `exp build PROJECT [-t PATH] --source SOURCE --root ROOT [--provider NAME ...]` | Launch the guided end-to-end build when traces are omitted, or use one explicit local source for automation. | Simulation, serving RAG, fit RAG, syllabus, evaluation evidence, and a runnable automatic router. |
| `exp optimize router PROJECT --root ROOT [--yes]` | Complete bounded simulation and judgment, fit a frozen router, then verify held-out evidence. | Fit evaluation, policy, held-out evaluation, and router report. |
| `exp optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `exp run --root ROOT [--check]` | Validate or start the initialized authenticated multi-alias gateway on loopback. | OpenAI-compatible endpoint, readiness routes, and content-free usage view. |
| `exp run PROJECT --root ROOT [--ghost]` | Activate a frozen policy as one project-backed alias and launch the normal gateway. | The same authenticated OpenAI endpoint and SQLite accounting as no-argument `exp run`. |
| `exp config gateway ...` | Author provider references, identities, virtual keys, grants, aliases, certified exact-model pools, monthly limits, status, and usage without optimizer roles. | Private SQLite authority, immutable catalog snapshots, and versioned receipts. |
| `exp config providers [--provider NAME ...]` | Collect secret-free provider connections, model aliases, and build roles. | Local `.exp/models.toml`. |
| `exp config budget [USD] --root ROOT` | Read or set the maximum conservative estimate allowed for one paid command. | Local `.exp/settings.toml`. |
| `exp config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.exp/settings.toml`. |

`build`, judge calibration, `optimize router`, and `optimize model` use the same cost authorization
policy. An estimate at or below 50% of the budget runs automatically. A higher estimate
up to the budget requires a clear terminal confirmation or `--yes`; an estimate above the budget
warns and requires an explicit interactive override that defaults to no, and fails closed before
credentials or provider clients when no terminal is available. Set the deterministic ceiling with
`exp config budget USD --root ROOT`. `--yes` confirms only an in-budget invocation and never raises
the ceiling. Exact completed replays report a zero-dollar estimate and do not prompt.

Successful build, router, simulation, and SFT operations preserve anonymous aggregate PostHog
product telemetry, which may send unless disabled. `run` makes no provider call at startup.
`build --dry-run` and exact completed-build replay make zero paid provider calls. A new grounded
build calls only the configured embedder; automatic router optimization separately executes the
bounded candidate, world-model, and judge schedule shown in its cost preflight.
An authenticated gateway request is the explicit online model-call boundary. Project selectors
remain frozen for the process lifetime and return only an exact model pool. `--ghost` remains a
compatibility flag for project-journal behavior; gateway authentication, replay, attempts, and
usage accounting stay enabled.

Both `exp run` forms use one gateway lifecycle. It binds only `127.0.0.1`, starts with no
provider call, and requires an explicit provider environment reference, exact model alias, identity,
grant, and virtual key. `exp run --non-interactive --json` returns `gateway_not_initialized` plus
exact next commands on an empty root. `exp run --check` validates local readiness without binding.
The gateway writes no prompts, responses, tool arguments, raw keys, or provider secrets to SQLite.
`GET /usage` and `GET /usage.json` expose the same schema-v2 content-free overall and per-identity
counts, token usage, latency, terminal states, and attributed estimated cost. Their attempt-only
`by_billing_source` buckets conserve attempts, tokens, known cost, unknown-cost attempts, and
terminal states across `host_managed` and `customer_managed`; logical request counts are not
partitioned. Estimated cost is not provider invoice cost.

One-time virtual-key material appears only in the successful key-issue receipt or a newly created
mode-`0600` output file. Human key issuance on a non-terminal requires `--json` or `--output`.
Provider configuration accepts an environment variable name, never a raw credential value. Its
current revisions live in SQLite; immutable serving snapshots bind exact revisions while build and
evaluation artifacts remain in the project artifact store.

To add ordered failover, first author each deployment as a direct alias with the same
`--exact-model`. Then certify their equivalence and order with:

```console
exp config gateway pool certify PUBLIC_ALIAS \
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

Direct deployments declare `--billing-source host_managed` or `customer_managed`. The selected
value is frozen on each physical attempt before dispatch and remains unchanged across catalog
replacement and restart. Usage JSON and HTML expose content-free physical-attempt buckets by source;
they do not partition logical request counts. Legacy schema-v1/v2 attempts migrate explicitly as
`customer_managed`.

Monthly serving limits are separate from the one-shot `exp config budget` command ceiling. They use
integer micro-USD and explicit immutable UTC periods. The local team, an identity, a total alias
pool, and each provider deployment can have overlapping hard limits:

```console
exp config gateway budget set --period 2026-08 --scope identity \
  --identity TEAM_MEMBER --limit-micro-usd 20000000000 \
  --root ROOT --non-interactive --json
exp config gateway budget set --period 2026-08 --scope deployment \
  --alias PUBLIC_ALIAS --pool EXACT_POOL --deployment AZURE_DEPLOYMENT \
  --limit-micro-usd 10000000000 --root ROOT --non-interactive --json
exp config gateway budget set --period 2026-08 --scope deployment \
  --alias PUBLIC_ALIAS --pool EXACT_POOL --deployment BEDROCK_DEPLOYMENT \
  --limit-micro-usd 10000000000 --root ROOT --non-interactive --json
exp config gateway budget remaining --period 2026-08 --root ROOT --json
```

Omit required values in an interactive terminal to receive prompts. Pass `--non-interactive` to
fail immediately instead. `--replace` changes a configured limit without deleting the month or its
spend. Exhausting one deployment continues to the next certified exact-model route. Exhausting all
applicable shared capacity returns OpenAI `insufficient_quota`. Unknown required pricing fails
closed under a hard limit. There is no budget reset job and no budgets dashboard.

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

The guided build uses one bounded consent to create simulation evidence, separate serving and fit
RAG indexes, a judge syllabus, closed-loop candidate evaluations, and a runnable router. Human
judge calibration is recommended but optional: provisional judgment provenance remains visible,
and later approval plus another build creates an immutable human-calibrated successor. Running the
endpoint records traffic by default so a later optimization can use newly attributed outcomes.
Explicit trace automation can still stop after the grounded build, then use
[`router-optimization.json`](reference/router_optimization_config.md) with `exp optimize router`.
Router fitting never invokes world-model fidelity testing. Applications that need a world-model
quality measurement can call the separate `build_fidelity_evaluation_plan` and
`build_fidelity_report` APIs; those results never enter router fitting or activation. Fidelity
reports contain measurements only, never an approval or denial. See the
[router contracts](reference/router_optimization_config.md).

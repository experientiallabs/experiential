# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `exp` | Open the branded home screen. `Run Gateway` is the first option and runs setup when needed. | Interactive gateway menu, or the default gateway in a non-interactive terminal. |
| `exp login [--root ROOT]` | Sign in to Experiential Cloud through the Platform browser approval flow, save the returned organization key, and synchronize the authenticated account's model identities. | User-local credential plus secret-free hosted provider/model records in `.exp/models.toml`. |
| `exp run [PROJECT] [--root ROOT] [--check]` | Start the local gateway directly, optionally with one project-backed alias. | OpenAI-compatible endpoint, readiness routes, and content-free usage view. |
| `exp build PROJECT [-t PATH] --source SOURCE --root ROOT [--provider NAME ...]` | Launch the guided end-to-end build when traces are omitted, or use one explicit local source for automation. | Simulation, serving RAG, fit RAG, syllabus, evaluation evidence, and a runnable automatic router. |
| `exp optimize router PROJECT --root ROOT [--yes]` | Complete bounded simulation and judgment, fit a frozen router, then verify held-out evidence. | Fit evaluation, policy, held-out evaluation, and router report. |
| `exp optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `exp --root ROOT [--check]` | Validate or start the initialized authenticated default gateway on loopback; the native data plane serves every route, including Chat Completions, Responses, and Anthropic Messages. | OpenAI-compatible and Anthropic Messages endpoints, readiness routes, and content-free usage view. |
| `exp --project PROJECT --root ROOT [--ghost]` | Activate a frozen policy as one project-backed alias and launch the normal gateway. | The same authenticated OpenAI endpoint and SQLite accounting as the default gateway. |
| `exp config gateway ...` | Author provider references, identities, virtual keys, grants, aliases, certified exact-model pools, monthly limits, status, and usage without optimizer roles. | Private SQLite authority, immutable catalog snapshots, and versioned receipts. |
| `exp config gateway call ALIAS PROMPT [--json]` | Send one chat completion to a live gateway as a caller, streaming text to stdout. | One HTTP request against the running gateway; no local state. |
| `exp config gateway models [--json]` | List the aliases a live gateway grants to the presented key (caller view of `GET /v1/models`). | One HTTP request against the running gateway; no local state. |
| `exp config gateway key check [--json]` | Validate one raw virtual key against a live gateway and print its granted aliases without storing the key. | One HTTP request against the running gateway; no local state. |
| `exp config providers [--provider NAME ...]` | Collect secret-free provider connections, model aliases, and build roles. `experiential-cloud` points at the hosted Platform gateway and reuses the credential from `exp login`; login already performs its provider/model synchronization. Setup also persists, replaces, or removes user-local provider keys. | Local `.exp/models.toml` plus optional records in the user-data credential file. |
| `exp config budget [USD] --root ROOT` | Read or set the maximum conservative estimate allowed for one paid command (default `$50.00`). | Local `.exp/settings.toml`. |
| `exp config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.exp/settings.toml`. |

`build`, judge calibration, `optimize router`, and `optimize model` use the same cost authorization
policy. An estimate at or below 50% of the budget runs automatically. A higher estimate
up to the budget requires a clear terminal confirmation or `--yes`; an estimate above the budget
warns and requires an explicit interactive override that defaults to no, and fails closed before
credentials or provider clients when no terminal is available. Set the deterministic ceiling with
`exp config budget USD --root ROOT`. `--yes` confirms only an in-budget invocation and never raises
the ceiling. Exact completed replays report a zero-dollar estimate and do not prompt.

Successful build, router, simulation, and SFT operations preserve anonymous aggregate PostHog
product telemetry, which may send unless disabled. Gateway startup makes no provider call.
`build --dry-run` and exact completed-build replay make zero paid provider calls. A new grounded
build calls only the configured embedder; automatic router optimization separately executes the
bounded candidate, world-model, and judge schedule shown in its cost preflight.
An authenticated gateway request is the explicit online model-call boundary. Project selectors
remain frozen for the process lifetime and return only an exact model pool. `--ghost` remains a
compatibility flag for project-journal behavior; gateway authentication, replay, attempts, and
usage accounting stay enabled.

The default and project gateway forms use one gateway lifecycle. It binds only `127.0.0.1`, starts with no
provider call, and requires an explicit provider environment reference, exact model alias, identity,
grant, and a virtual key. Interactive first-run setup can persist multiple provider connections and
creates one initial gateway alias; additional deployments and certified pools remain explicit
gateway configuration. From the interactive home screen, `Setup Gateway` also offers a confirmation-gated
reconfiguration path for an initialized gateway: it replaces the selected provider and alias revisions
while preserving existing identities, keys, grants, usage, and history. `exp --non-interactive --json` returns `gateway_not_initialized` plus exact
next commands on an empty root. `exp --check` validates local readiness without binding.
First-run setup prints `EXP_GATEWAY_URL` and the newly issued `EXP_GATEWAY_KEY` before readiness is
checked, so the credentials remain available even when a provider route is not ready. The gateway-
specific variables avoid overwriting an upstream provider's `OPENAI_API_KEY`. The error names the
unavailable alias and provider configuration; fix that configuration and rerun `exp`. If the
one-time key was not saved, issue a replacement with
`exp config gateway key issue IDENTITY --key-id KEY --json`.
The gateway writes no prompts, responses, tool arguments, raw keys, or provider secrets to SQLite.
`GET /usage` and `GET /usage.json` expose the same schema-v2 content-free overall and per-identity
counts, token usage, latency, terminal states, and attributed estimated cost. Their attempt-only
`by_billing_source` buckets conserve attempts, tokens, known cost, unknown-cost attempts, and
terminal states across `host_managed` and `customer_managed`; logical request counts are not
partitioned. Estimated cost is not provider invoice cost.

One-time virtual-key material appears only in the successful key-issue receipt or a newly created
mode-`0600` output file. Human key issuance on a non-terminal requires `--json` or `--output`.
Provider catalogs and gateway SQLite store an environment-variable name, never a raw credential
value. Interactive `exp config providers` persists a pasted key in the platform user-data file
(`~/.local/share/exp/auth.json` on Linux) and can replace or remove that stored key when the
same provider is edited again. Runtime commands never prompt. They resolve an explicit
caller-supplied environment mapping first, then a non-empty process environment value, then
the stored key for that connection ID. Environment values override the store without rewriting
it. Missing credentials fail with the environment name and a recovery that points at
`exp config providers`. Bedrock stays on the AWS credential chain. Current provider revisions
live in SQLite; immutable serving snapshots bind exact revisions while build and evaluation
artifacts remain in the project artifact store.

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
applicable shared capacity returns OpenAI `insufficient_quota`. By default an unpriced attempt is
admitted and recorded as unknown cost; `budget remaining` reports the unknown-cost attempts and
their observed token volume. `--strict-unknown-cost` on `budget set` opts one limit into failing
closed instead: unpriced attempts are rejected, recorded unknown-cost attempts block the limit even
after `--replace` raises it, and `exp config gateway budget reconcile --period 2026-08 --scope
team --assigned-cost-micro-usd COST --root ROOT --non-interactive` settles each unknown-cost
attempt at an explicit assigned cost and restores service with exact per-attempt attribution.
There is no budget reset job and no budgets dashboard.

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

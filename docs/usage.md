# CLI usage

Gateway embedders can use the [capture hosting interface](reference/gateway_capture.md)
with their own consent policy and storage destination.

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `exp` | Open the branded home screen. `Run Gateway` is the first option and runs setup when needed. | Interactive gateway menu, or the default gateway in a non-interactive terminal. |
| `exp login [--root ROOT]` | Sign in to Experiential Cloud, save the returned organization key, and synchronize account-visible models with the catalog's default-route capabilities and undiscounted prices. Known metadata is reused without capability or price questions. | User-local credential plus secret-free hosted provider/model records in `.exp/models.toml`. |
| `exp run [PROJECT] [--root ROOT] [--check]` | Start the local gateway directly, optionally with one project-backed alias. | OpenAI-compatible endpoint, readiness routes, and content-free usage view. |
| `exp eval [PROJECT] --models ALIAS,ALIAS` | Compare models on the project scenarios, or open the terminal project picker. | Saved resumable run, JSON evidence, and offline Pareto report. |
| `exp build PROJECT [-t PATH] --source SOURCE --root ROOT [--provider NAME ...]` | Import file or gateway traces, mine scenarios, and prepare world-model grounding; omitting traces opens the guided build. | Canonical imports in `gateway/traffic.db`, versioned scenarios, serving RAG, fit RAG and a grounded world model. |
| `exp optimize router PROJECT --root ROOT [--yes]` | Complete bounded simulation and judgment, fit a frozen router, then verify held-out evidence. | Fit evaluation, policy, held-out evaluation, and router report. |
| `exp optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `exp --root ROOT [--check]` | Validate or start the initialized authenticated default gateway on loopback; the native data plane serves every route, including Chat Completions, Responses, and Anthropic Messages. | OpenAI-compatible and Anthropic Messages endpoints, readiness routes, and content-free usage view. |
| `exp --project PROJECT --root ROOT [--ghost]` | Activate a frozen policy as one project-backed alias and launch the normal gateway. | The same authenticated OpenAI endpoint and SQLite accounting as the default gateway. |
| `exp config gateway ...` | Author provider references, identities, virtual keys, grants, aliases, certified exact-model pools, monthly limits, status, and usage without optimizer roles. | Private SQLite authority, immutable catalog snapshots, and versioned receipts. |
| `exp config gateway call ALIAS PROMPT [--json]` | Send one chat completion to a live gateway as a caller, streaming text to stdout. | One HTTP request against the running gateway; no local state. |
| `exp config gateway models [--json]` | List the aliases a live gateway grants to the presented key (caller view of `GET /v1/models`). | One HTTP request against the running gateway; no local state. |
| `exp config gateway key check [--json]` | Validate one raw virtual key against a live gateway and print its granted aliases without storing the key. | One HTTP request against the running gateway; no local state. |
| `exp config providers [--provider NAME ...]` | Collect secret-free provider connections, model aliases, and build roles. `experiential-cloud` points at the hosted Platform gateway and reuses the credential from `exp login`; login already performs its provider/model synchronization. Setup also persists, replaces, or removes user-local provider keys. | Local `.exp/models.toml` plus optional records in the user-data credential file. |
| `exp config budget [USD] --root ROOT` | Read or set the budget warning threshold for one paid command (default `$50.00`). | Local `.exp/settings.toml`. |
| `exp config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.exp/settings.toml`. |

`build`, `eval`, judge calibration, `optimize router`, and `optimize model` use the same cost authorization
policy. An estimate at or below 50% of the budget runs automatically. A higher estimate
requires a clear terminal confirmation or `--yes`. An estimate above the budget warns and offers
"Proceed anyway?", defaulting to no. `--yes` also authorizes an over-budget estimate after the
warning. Without a terminal or explicit consent, the command explains how to proceed and makes
no provider calls. Set the warning budget with `exp config budget USD --root ROOT`. Build embedding
and router budgets and the judge-calibration budget use the same warning and confirmation flow.
Approval applies to this invocation; saved budgets stay unchanged. Exact completed replays report
a zero-dollar estimate and do not prompt.

Successful build, router, simulation, and SFT operations preserve anonymous aggregate PostHog
product telemetry, which may send unless disabled. Gateway startup makes no provider call.
`build --dry-run` and exact completed-build replay make zero paid provider calls. A new grounded
build calls only the configured embedder; automatic router optimization separately executes the
bounded candidate, world-model, and judge schedule shown in its cost preflight.
An authenticated gateway request is the explicit online model-call boundary. Project selectors
remain frozen for the process lifetime and return only an exact model pool. `--ghost` disables
local traffic content capture; gateway authentication, replay, attempts, and usage accounting
stay enabled.

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
The accounting database stays content-free. Local traffic content is captured separately by default;
use `--ghost` to disable it. See [local traffic capture](reference/local_gateway_traffic.md).
Raw virtual keys and resolved provider credentials are never copied into capture or accounting.
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
`exp config providers`. The model picker accepts multiple models from each provider (Space toggles
selections). Selected models stay in the catalog for evaluation even without a build role.
Reasoning choices come from the selected deployment or maintained provider contract; DeepSeek
shows its distinct off (`none`), low, high, and max modes instead of compatibility aliases.
Bedrock stays on the AWS credential chain. Current provider revisions
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
integer nano-USD (a billionth of a dollar; `20000000000000` is $20,000) and explicit immutable UTC
periods. The local team, an identity, a total alias
pool, and each provider deployment can have overlapping hard limits:

```console
exp config gateway budget set --period 2026-08 --scope identity \
  --identity TEAM_MEMBER --limit-nano-usd 20000000000000 \
  --root ROOT --non-interactive --json
exp config gateway budget set --period 2026-08 --scope deployment \
  --alias PUBLIC_ALIAS --pool EXACT_POOL --deployment AZURE_DEPLOYMENT \
  --limit-nano-usd 10000000000000 --root ROOT --non-interactive --json
exp config gateway budget set --period 2026-08 --scope deployment \
  --alias PUBLIC_ALIAS --pool EXACT_POOL --deployment BEDROCK_DEPLOYMENT \
  --limit-nano-usd 10000000000000 --root ROOT --non-interactive --json
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
team --assigned-cost-nano-usd COST --root ROOT --non-interactive` settles each unknown-cost
attempt at an explicit assigned cost and restores service with exact per-attempt attribution.
There is no budget reset job and no budgets dashboard.

## Standalone model evaluation in Python

Rollouts default to 100 candidate steps and 1,000,000 cumulative candidate output tokens.
Both are configurable without a fixed engine step ceiling. `maximum_output_tokens` is a
separate optional per-request setting; omitting it uses each model's declared output capacity.
Explicit limits are clamped to that capacity
and, for candidates, the remaining rollout token budget. Provider output usage includes
reasoning; it is not counted twice. Missing usage blocks further candidate dispatch.

Budget exhaustion and truncated output are `incomplete`, excluded from judging, quality and
operating-cost comparisons. Actual incurred spend remains in execution accounting. A complete,
secret-free text-world turn saves a checkpoint. To continue the built-in chat runtime, prepare
another evaluation with `continuation_of=previous.simulation_spec.simulation_id` and larger
`ModelEvaluationOptions(maximum_steps=200, maximum_rollout_output_tokens=2_000_000)`.
Keep the same workers, judge setup/calibration, prompts, retrieval, per-call reservations and
producer revision; then authorize its quote with `run_prepared_model_evaluation` as usual.
This creates a new immutable execution and parent-linked rollouts. Completed candidate/world
work is retained without redispatch; cumulative costs include the retained prefix. Exact replay
of the child also reuses its judgments. Continuation does not restore arbitrary custom-agent
process state, truncated generations, interrupted world turns, or redacted transcript content;
those require a fresh evaluation. No prior artifact is edited.

Declared tools run against generated environment observations, not real external tool
implementations. A safe checkpoint preserves ordered tool results and private world state;
continuing does not re-execute completed tool turns. The retrieval estimate assumes one query
per turn, while its maximum reserves for multiple tool calls using each worker's output limit.

For a completed grounded project, `exp.prepare_model_evaluation` freezes a worker matrix and
prices its simulation, retrieval and judge requests without calling providers. The default judge
is binary task success with explicitly provisional provenance. Pass both `judge_setup` and
`calibration_id` to use a saved authored or calibrated judge instead. No router is fitted or
activated.

```python
from datetime import UTC, datetime

from exp import (
    EvaluationBudget,
    ModelEvaluationOptions,
    prepare_model_evaluation,
    run_prepared_model_evaluation,
)

# project is a completed ProjectStore; catalog is its secret-free ModelCatalog.
prepared = prepare_model_evaluation(
    project,
    catalog,
    ("worker-a", "worker-b"),
    embedder_alias="embedder",
    options=ModelEvaluationOptions(maximum_steps=8),
    created_at=datetime.now(UTC),
    code_revision=engine_revision,
)

# Display prepared.cost. A host reserves sufficient credits atomically and obtains
# consent before constructing its credential-backed runtime_catalog and executing.
result = run_prepared_model_evaluation(
    project,
    prepared,
    runtime_catalog,
    budget=EvaluationBudget(
        maximum_cost_usd=prepared.cost.maximum_cost_usd,
        maximum_judgments=prepared.cost.judgment_count,
    ),
    provider_spend_consented=consent_after_credit_reservation,
    created_at=run_started_at,
    code_revision=engine_revision,
)
```

`result.report` contains the common-cohort model metrics, explicit exclusions and cost-quality
frontier. The chart's cost is worker operating cost, not the cost of generating the report.
`result.cost_usd` reconciles simulation and judging charges. Exact replay dispatches no new model
calls. The quote and result exclude earlier trace mining and grounding costs; a host must include
those separately before offering a complete trace-to-report price. Credit conversion, promotions,
identity authorization and job persistence remain hosting responsibilities.

## Gateway clients

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

The guided build defaults to importing traces, mining scenarios, and creating separate serving
and fit RAG indexes with world-model grounding. Judge editing, calibration, and router optimization
are explicit optional steps. Selecting router optimization uses one bounded consent for the
combined build and evaluation schedule. Human judge calibration is recommended but optional:
provisional judgment provenance remains visible, and later approval creates an immutable
human-calibrated successor. Running the endpoint records traffic by default so a later optimization
can use newly attributed outcomes. Completed grounded projects can use
[`router-optimization.json`](reference/router_optimization_config.md) with `exp optimize router`.
Router fitting never invokes world-model fidelity testing. Applications that need a world-model
quality measurement can call the separate `build_fidelity_evaluation_plan` and
`build_fidelity_report` APIs; those results never enter router fitting or activation. Fidelity
reports contain measurements only, never an approval or denial. See the
[router contracts](reference/router_optimization_config.md).

### Simulated tool observations

Text world models generate environment observations for declared tools. Candidate tool calls are
passed to the world model together with schemas, the visible transcript, retrieved trace examples,
and private environment state. Parallel calls receive one ordered tool message per call ID; no
external tool implementation runs. Malformed observation batches are invalid rollouts.

The public session API accepts `world.new_session(task="Research", tools=(tool_schema,))` followed
by `world.step(session.id, assistant_message)`. Each result exposes `messages`, an ordered tuple
of OpenAI user or tool messages, and `terminal`. Tool observations are nonterminal so the agent can
consume them before producing its final answer. World-model artifacts pin the v2 prompt; rebuild
projects created with a different prompt before running them.

### Build scenarios from traces

```bash
exp build powerset --traces rollouts.jsonl --source chat-json --root .exp
exp build powerset --source gateway --identity default --root .exp
```

Build first stores canonical traces in `.exp/gateway/traffic.db`, alongside native gateway captures.
It preserves initial system/developer instructions, declared tool schemas, paired tool results,
source provenance, normalization exclusions and model identity evidence. The receipt identifies
an immutable import associated with the project. Repeating an unchanged import reuses its records
and association; changed source content produces a new import without overwriting earlier evidence.

It then mines scenarios from that saved evidence, writes immutable task sets and prepares
world-model grounding. Model roles belong to the project; embedding work uses the normal cost
preflight. An unchanged completed build reuses its scenarios and indexes without new provider calls.
`--dry-run` uses temporary SQLite for source ingestion and may checkpoint deterministic project
evidence, but makes no provider calls, durable trace imports, or completed-build selection.
The interactive build prompts for an explicit source file and detects chat JSON, native capture,
and recognizable OpenTelemetry exports. Unknown or ambiguous files get a format question in the
TUI. An explicit `--source` is always respected; automation supplies `--traces` and `--source`.
OTel sources (`otlp`, `otel-genai`) and completed exported chat captures (`experiential`) use the
same persistence path. See [trace input and storage](reference/ingest.md) for the Python API.

Native capture exports must contain completed JSON Chat Completions responses. Export stream
captures as reconstructed `chat-json`, or use `--source gateway` to consume the native database's
JSON/SSE captures. Incomplete or refused exports appear as explicit exclusions. Chat exports
without timestamps retain synthetic ordering markers and do not imply measured latency.
There is no minimum or maximum import count; a build requires at least one valid trace.
Source ingestion streams through disk-backed staging. Scenario mining and RAG preparation
currently materialize the canonical corpus, so the whole build's memory grows with that corpus.

### Local gateway traffic

See [local traffic capture](reference/local_gateway_traffic.md) for default-on,
identity-scoped collection. `--source gateway` reads every retained record for the required
identity from one consistent snapshot, including corpora larger than one page.

### Evaluate a project

`exp eval powerset --models gpt-5.6-luna,deepseek-v4.1-flash` prepares and reviews the model matrix,
then simulates, judges, and writes a report. Model names are configured catalog aliases; project
world-model and judge choices come from the configured project. Existing authored/calibrated project judges
are reused by default. Choosing another judge model preserves the project rubric and prompt,
creates provisional calibration for that run, and leaves project defaults and previous runs intact.
Without an authored judge, the task-success judge is explicitly provisional.

`exp eval powerset` requires a completed `exp build powerset` first. It opens a small project
screen: select models, runs per scenario, and judge; review the estimate; then choose Start
evaluation. Provider connections, traces, scenarios, and world-model grounding come from build.
Advanced rollout budgets are optional. Saved results show score, assistant cost, and latency;
Open report opens plots and side-by-side traces. Details exposes paths and accounting.
The default minimum is 20 distinct scenarios,
with one repeat, eight parallel workers, 100 steps, and 1,000,000 generated tokens per rollout.
Retries are separate from repeats. New evaluations collect fresh evidence; resume reuses the
exact saved run. Settings are saved in the project's `evaluation.json`.

Use `--dry-run` for provider-free preparation, `--resume RUN_ID` for exact saved execution, and
`--report RUN_ID` for read-only results. Ctrl-C cancels queued work and drains active rollouts;
completed cells remain immutable and resume does not repeat them. Unknown in-flight provider
outcomes remain invalid evidence, with their cost reservation retained by the engine.

Each completed run writes `report.json`, `rollouts.jsonl`, and a standalone offline `report.html`
under `.exp/projects/PROJECT/runtime/evaluations/RUN_ID/`. The HTML contains a cost-quality Pareto
plot and one tile per scenario with model and repeat selectors. Assistant cost and quality use the
shared valid cohort; invalid/incomplete coverage and total experiment spend remain visible.

Assistant cost per task reprices recorded successful-rollout tokens at the frozen catalog rates.
It excludes simulation, judging, invalid attempts, and hypothetical retry reservations.
Conservative experiment-spend accounting remains separate from the report's operating cost.

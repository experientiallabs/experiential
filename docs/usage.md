# CLI usage

Gateway embedders can use the [capture hosting interface](reference/gateway_capture.md)
with their own consent policy and storage destination.

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `exp` | Open the branded home screen. `Run Gateway` is the first option and runs setup when needed. | Interactive gateway menu, or the default gateway in a non-interactive terminal. |
| `exp login [--root ROOT]` | Sign in to Experiential Cloud, save the returned organization key, and synchronize account-visible models with the catalog's default-route capabilities and undiscounted prices. Known metadata is reused without capability or price questions. | User-local credential plus secret-free hosted provider/model records in `.exp/models.toml`. |
| `exp capture [--verbose] [--domain HOST ...]` | Capture supported OpenAI and Anthropic traffic across macOS apps until Ctrl+C, reusing `exp login`. | Cloud traces and bounded private retry batches. |
| `exp run [PROJECT] [--root ROOT] [--check]` | Start the local gateway directly, optionally with one project-backed alias. | OpenAI-compatible endpoint, readiness routes, and content-free usage view. |
| `exp eval [PROJECT] --models ALIAS,ALIAS` | Compare models on the project scenarios, simulated with the build's learned memory, or open the terminal project picker. | Saved resumable run, JSON evidence, and offline Pareto report. |
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

## Direct provider capture on macOS

Capture is experimental. Live Codex capture and recovery from a crashed or frozen Capture
process have been exercised on macOS; broader application and network compatibility still needs testing.

Run `exp capture` in a terminal and leave it open while using your AI applications. It captures
supported traffic across apps using `api.openai.com`, `chatgpt.com`, and `api.anthropic.com`,
including Codex and Claude Code, without a provider picker or changing provider base URLs. Capture
reuses the normal Experiential login, or opens the same login flow when no credential exists. The Capture
tab under API Keys shows the run and its statistics for Platform administrators. A run represents
one foreground Capture process, rather than one application conversation. No separate capture
credential or background capture daemon is required.
Only one Capture process can run per macOS user, including across different preview profiles.

The terminal shows setup progress, live capture counts, and provider-reported input/output tokens.
Token totals use K, M, B, and T with up to two decimals, such as 1.75M for 1,752,000 tokens.
Totals include cached input; missing usage is marked partial or unavailable. Use
`exp capture --verbose` (or `-v`) for timestamped TLS and request events, upload counters, provider
hosts, and the public CA certificate path. Verbose receipts link each request to its saved trace
and upload batch, with a hashed provider response ID, reported tokens, and separate completion
and transport-error indicators. Upload acceptance does not confirm dashboard processing.
Slow verbose output can drop diagnostic events, with a count when output resumes; trace storage
and delivery continue independently.
Approval requests and errors remain visible without verbose output. When macOS reports the
extension awaiting approval, Capture opens Login Items & Extensions and continues once you enable
Mitmproxy Redirector under Network Extensions.

Capture requires Python 3.13 or newer. Other SDK and CLI commands continue to support Python 3.12.
In a checkout, use `uv run --python 3.13 exp capture`. The pinned fork dependencies may require
Rust for the first source build. The first run installs the bundled, signed
Mitmproxy Redirector app at `/Applications/Mitmproxy Redirector.app`. Approve its Network Extension
when macOS asks. Before login, Capture verifies the packaged app and extension signatures against
mitmproxy's expected signing identity, verifies any installed copy it will reuse, and checks the
supported macOS version and installation access. Identical installed app contents are reused
across Python environments without replacing the app or requesting new setup approval.
If your account cannot install or update the app
in `/Applications`, ask your administrator for installation access, then retry as your normal user.

Advanced users can repeat `--domain HOST` to replace the defaults with an exact set, for example
`exp capture --domain api.openai.com`. The filter applies across applications. Only supported
model request paths produce uploaded traces; authentication and unrelated web requests are
forwarded without retaining their bodies.
Captured traces include prompts, responses, and tool content. Incomplete or malformed structured
tool arguments are replaced with a redaction marker. Credential headers are never
copied into uploaded traces. This is separate from anonymous aggregate product telemetry.

The first run creates a private local certificate authority with critical X.509 name constraints
for exactly the selected provider hostnames and requests SSL trust for the current macOS user.
The certificate excludes subdomains and IP addresses. Selecting both a parent hostname and one
of its subdomains is rejected because those exact-host constraints would conflict. The native
macOS and Chromium verifiers enforce the certificate constraints; hostname-specific macOS trust
settings are not used because Chromium does not support them.

Each provider-host set gets a separate CA under `capture/ca-constrained/<scope-hash>` in the same
user-data directory that owns the saved login. `exp capture --verbose` prints its public
`mitmproxy-ca-cert.pem` path. A client with its own trust store may need that public CA configured
explicitly; certificate pinning is not bypassed. Never share the adjacent `mitmproxy-ca.pem`, which
contains the private signing key. Existing unconstrained certificates are not reused or granted
broader trust.

If a client explicitly rejects the Capture certificate, its subsequent connections to that host
pass through with the original certificate and are not captured for the rest of the run. Retry or
reload the affected app after the first failed connection. Capture names the app and host, keeps
partial coverage visible in the terminal, and continues capturing other apps and hosts. If native
process identity is unavailable or the bounded app list is full, all apps pass through for that
host, with the wider exclusion shown explicitly. No client verification or certificate pinning is
disabled. Ambiguous handshake disconnects remain local to the failed connection and do not
turn off capture for future requests. Native write failures also close only the affected connection.
Verbose diagnostics include fixed TLS, request, and DNS-forwarding lifecycle categories, opaque
connection IDs, and DNS-check timeouts or resolver exit codes, without request
content, authentication headers, URL queries, or raw error strings.
If the native capture backend itself exits, Capture stops and reports that failure.

Capture quietly checks provider DNS before starting and every few seconds while running.
Two consecutive failures for the same provider stop interception automatically. A final check
reports whether DNS recovered; Capture does not restart itself or request additional permissions.
The macOS DNS responder process is excluded from interception. DNS attributed to other apps can
still traverse the redirector, so this guard detects resolver failures rather than guaranteeing
uninterrupted application connections. It does not change DNS settings or reset system services.

Interception starts only after an independent watchdog is ready. If the foreground process dies
or its event loop stops sending heartbeats for 15 seconds, the watchdog disables interception
and terminates a stalled owner. macOS UDP sockets remain open across idle periods until their
application closes them or Capture stops. The pinned proxy forks provide that lifetime and
cleanup behavior while retaining the original signed macOS extension.

Run the CLI as your normal user, without `sudo`. Mitmproxy Redirector's Network Extension provides
system-wide interception while the foreground backend is running. Ctrl+C ends that interception;
Capture does not edit `/etc/hosts`, DNS settings, or system proxy settings. There is no reset
command or separate Capture service to stop. The signed app and its macOS approval remain
installed, and the local CA trust remains in the current user's trust store between runs.
Existing intercepted connections may close when Capture stops. Restart an application if its
existing connections prevent a new run from observing requests. Certificate-pinned apps and
unsupported model protocols are not collected. UDP traffic, including QUIC/HTTP3 and DNS,
passes through without inspection; model trace collection currently supports HTTPS over TCP.

Upload failures do not stall model responses. The collector retains bounded private retry files
under the origin-and-organization-specific `capture/spool` directory. The next capture run with
the same endpoint and organization retries pending files using their original batch IDs.
When capture capacity is exhausted, collection drops copies and reports the count while model
traffic continues. The CLI distinguishes captured requests, uploaded batches, and pending batches;
upload acceptance does not imply that cloud projection has finished.
Each run pins the storage origin and organization path configured by Platform. Upload tickets
pointing outside that destination are rejected. Platform remains the trusted recipient and
control plane for captured content.

For an unreleased Platform preview, set both `EXP_GATEWAY_URL` to the preview API `/v1` URL and
`EXP_PLATFORM_URL` to its web origin before `exp login` and `exp capture`. Saved credentials are
bound to their endpoint, so a production login is not silently sent to a preview. If the saved
login belongs to another endpoint, Capture opens normal login for the configured environment;
successful login replaces the saved CLI login. To keep a preview's login and catalog separate,
set `XDG_DATA_HOME` to a dedicated preview data directory and pass a separate `--root` to Capture.
Capture checks the cloud API before starting interception. Live Codex and Claude Code
compatibility remains part of local acceptance; existing open connections may need to be
restarted to enter capture.

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
The accounting database stays content-free. Local traffic content is captured separately by default
into a plaintext SQLite file under the root (the launch receipt names it as `traffic_database`);
use `--ghost` to disable it. See [local traffic capture](reference/local_gateway_traffic.md).
`exp` reads a `.env` file in the current directory at startup, so provider keys placed there are
visible to every gateway connection that names them; keep secrets elsewhere if that directory is
shared. Aggregate PostHog telemetry is on until `exp config telemetry disable`.

`exp run --request-timeout SECONDS` sets the total budget for one request, failover included
(default 120, at most 3600). A streamed answer cut at that limit ends with an explicit error event
naming the time limit (Chat: an error chunk then `[DONE]`; Messages: an `error` event with no
`message_stop`; Responses: `response.failed`), never a stream that simply stops, so a client can
tell a cut answer from a finished one.
Raw virtual keys and resolved provider credentials are never copied into capture or accounting.
`GET /usage` and `GET /usage.json` expose the same schema-v3 content-free overall and per-identity
counts, token usage, latency, terminal states, and attributed estimated cost. Their attempt-only
`by_billing_source` buckets conserve attempts, tokens, known cost, unknown-cost attempts, and
terminal states across `host_managed` and `customer_managed`; logical request counts are not
partitioned. `by_client_app` attributes each request to the calling app (Claude Code, Codex,
OpenCode, Hermes Agent, and the rest of `exp.runtime.gateway.client_apps.ClientApp`), classified
from the `User-Agent`, Codex `originator`, `X-Title`, and `HTTP-Referer` headers. A caller no
app rule names is reported by its caller kind (OpenAI, Anthropic or Vercel AI SDK, browser, curl,
or custom code); only one with no recognizable header reports as unidentified. Estimated cost is not provider invoice cost.

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
        maximum_cost_usd=None,  # Optional aggregate cap; use a positive USD amount to enable one.
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

Authored (written or imported) scenarios: see `reference/authored-scenarios.md`.

Catalog-backed evaluations and `exp eval` have no aggregate spending limit by default. The
estimate remains visible and uncapped execution requires an explicit Start action or `--yes`.
Every request keeps finite token bounds and a retry-inclusive cost
reservation; completed charges, unknown-dispatch holds and exact response replay remain durable
with or without an aggregate cap. In the launch review, select Spending limit to set a positive
dollar amount or enter `none` to remove it. Changing that limit does not change the frozen plan.
`EvaluationServices` uses the execution budget when its spending limit is omitted; explicitly
passing `spending_limit_usd=None` disables only that aggregate limit, not a simulator's own
execution bound. The lower-level `evaluate_models` API still requires an explicit finite
simulation envelope. Use `run_prepared_model_evaluation` to derive that envelope automatically
from the catalog and frozen request limits while leaving aggregate spending uncapped.

Judges use their declared context capacity minus the output reservation. Full visible task and
tool evidence is retained; a transcript that exceeds that capacity is excluded with an explicit
admission reason. Judge requests preserve configured reasoning without adding sampling controls.
If a run stops during judging, select it under Saved evaluations in `exp eval PROJECT`. The launch
review offers a fresh judging pass over saved rollouts under the same spending limit. Successful
scores are reused, earlier failed-attempt spend is retained, and the new judging recipe is recorded
separately without editing the original run preparation or simulation artifacts. Request-ledger
charges include responses whose later probe or judgment write failed. The approved spending limit
governs recovery accounting independently of the original theoretical simulation ceiling.

## Local project state

`gateway/traffic.db` is the shared content database for captures, trace imports, and projects.
Project configuration has immutable versions and a current selection. Dataset/scenario versions,
judge definitions, judgments, and result manifests retain their exact IDs, lineage, and digests.
Evaluation runs pin configuration and prepared inputs; progress updates cannot rewrite them.
Checkpoint intents commit before provider dispatch so ambiguous completion never authorizes an
automatic duplicate call.

Embedding batches and evaluation request receipts use the same database. Embedding coordination
locks serialize duplicate build work without holding a SQLite transaction across provider calls.
Successful response artifacts settle atomically with request charges; unresolved evaluation calls
retain their conservative reservations. A new judging pass retains the previous completed receipt.

Large payloads remain digest-addressed files under `projects/PROJECT/blobs`, referenced by database
records. HTML reports remain generated exports. Copying a project folder alone is not a backup:
preserve a consistent SQLite backup and its referenced blobs together. Selected completed build
graphs can also be moved with the project bundle API. The bundle excludes active run state.

Build an imported corpus without rereading its source:

```console
exp build PROJECT --import-id IMPORT_ID --root ROOT
```

Folder layouts containing `project.toml` or `artifacts/` fail closed. Preserve them with their matching
release; export a verified project bundle there and restore it into a fresh root. No automatic
migration rewrites existing evidence.

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

### Learning from traces

`exp build PROJECT -t PATH` learns the simulator from the traces as part of every build. After
the index and simulator are built, it replays every trace once, in canonical dataset order (start time, then trace ID), on a hybrid
simulator: learned memory supplies behavioral rules, and the build's index supplies recorded facts
(a trace never retrieves its own lineage). Every step in a trace is predicted before its recorded
replies are disclosed and mismatches are investigated. Only investigations the investigator marks
transferable, phrased by the observable cue that triggers them, become memory rules; every
investigation stays recallable as a record. Each trace is published as one memory version and
the memory log is compressed. Memory records are recalled by a native decision model on the
SystemOne decisions wire: the catalog's `memory_recall` role when set, otherwise the catalog's
decision model, preferring Clef Flash (`cloudflare/clef-flash`), then Clef, then Jev. A catalog
without a decision model builds the simulator without learning and says how to add one. The pass is
resumable: completed traces replay without provider calls. Learning has no defined cost, so a
build that learns asks for consent at a terminal and needs `--yes` in automation; without it,
automation stops before any index is bought and prints the exact command to run again. The
learning memory belongs to the exact protocol (prompts, re-asks, token limits, attempts): changing
it learns a fresh memory trace by trace, while changing only the judge replays without calls.

Learning is measured test-then-train: each trace is predicted whole on the memory pinned before
it. None of the trace's own feedback or investigations reaches its predictions; they are
published with the trace. Memory from earlier traces of the same task lineage is hidden too,
just as retrieval hides the trace's own lineage. Memory written after a prediction saw retrieved
evidence depends on the retrieved traces, so a retrieved fact never reaches its own task through
memory either. The recorded actions are replayed in order, but after the trace's public prefix
the simulator sees its own earlier replies and state, never the recorded replies, so an early
error carries forward into later turns. Every learning report is therefore a measurement of
recorded actions over the simulator's own history, on a task memory has not learned, even though
every trace is consumed. The pass prints this as learning progress (replies exact and reply
fields matching), a learning curve of exact replies by trace order, and the latest unseen-task
estimate: the last quarter of traces, predicted on nearly final memory, with its trace count.
Traces whose learning fails remain in the score; their separate count shows when prediction
progress exceeds memory growth.

Learning progress averages simulators from empty memory to nearly final memory, so it shows
whether memory helps as it grows. It does not measure the final simulator on unseen traces:
after the pass, no trace is left unseen. Claims about the final simulator on unseen traces
come from `exp eval PROJECT --audit N` (below). The audit measures a separate simulator that
learned only the fit traces, a conservative proxy for the one learned from every trace; the
build's own all-traces simulator is not independently evaluated.

A simulator proposal that fails validation is never repaired: it is asked again, at most twice,
with the refusal reason, and each re-ask is saved and counted. After that the turn fails. A trace
buys at most 24 investigations across all feedback batches; failed investigation intents count
toward that limit too. Later misses are recorded as raw feedback without one, and the trace is
still learned. Concurrent investigations share one session lease, preventing another feedback
batch or prediction from changing the session while their receipts settle.

Each investigation sees the causal history of its trace: every step up to and including the
investigated one, with the public action, the simulator's predicted reply and committed state
edits, and the recorded reply disclosed for that step. Recorded replies are labelled native
observations and predictions are labelled the simulator's own belief, so a readback miss can be
attributed to the earlier write that explains it. Later steps and other traces never enter the
history. When a long trace exceeds the history budget, the oldest whole steps are dropped and the
request states how many were omitted; no step is cut partway.

A memory version pins its overview on first read; later compression cannot change it. Long rules keep both cue and claim as labelled previews, linked to their full records.
Learning banks bind the simulator's knowledge, the learning protocol and the scope. Changing a
serving setting (the simulator model, retrieval top-k) keeps compatible memory under a new
execution plan; changing knowledge or the learning protocol starts a fresh bank. A learning pass
never re-predicts a consumed trace on later memory. Fit-only banks remain separate. Earlier bank
contracts require a fresh project; they are not automatically migrated.

The build ends by retesting every learned trace that had at least one inexact reply, against
the final memory checkpoint, and compares each turn before and after learning: fixed, still
wrong, or regressed. The retest reads what memory learned from those
traces, so it checks that the lessons of a miss stuck; it does not measure unlearned tasks.

`exp eval PROJECT --models A,B` evaluates workers on the learned simulator, with no flag.
Preparation pins the build's memory at its latest version, and every simulated turn reads it:
records are recalled per tool call through the learning plan's recall alias and policy, and the
version's pinned overview comes with every turn. Evaluation only reads memory; it never publishes
or researches. Evaluation tasks are mined from the traces memory learned, so each task reads
memory with its own task lineage (and the lineages of its source traces) hidden, exactly as
learning hides it. Lineages reserved for evaluation stay withheld, and a held-out task reads only
memory of known provenance. The memory version is part of the evaluation plan: saved results
replay only for identical memory, and a run saved without memory resumes without it. The review
header and the saved report name the version (`Simulator memory: vN (...)`). A build that learned
nothing evaluates on the bare simulator and says so in one line. Recall decisions are saved and
replay on resume; they are not part of the spend estimate.

`exp eval PROJECT --audit N [--verbose]` is the independent audit. It first learns a separate
memory from fit traces only: held-out lineages are reserved before the first trace, so they are
never learned, and the fit-only index is used for recorded facts. Saved results replay without
calls. It then replays N held-out traces, selected deterministically, against that memory's
final checkpoint. It prints one aggregate line (replies exact, reply fields matching, judge
mean), how often the judge's per-turn verdict agrees with exact reproduction (a check on the
judge, not a score), and the learning curve. The report
names the audited checkpoint (memory lineage, version and `fit` scope). The build's all-traces
memory is a different checkpoint, which this audit does not evaluate independently.

Screened lessons. Learning records a miss on every inexact step; most describe one instance's
data, not how the environment behaves. To compare memory profiles on any project:

1. `exp eval PROJECT --audit N --audit-memory learned` learns the fit-only memory and audits it.
2. Screen its misses with `--continuations` and `{"action": "repair-learning", "learning_repair":
   {"resolution": "screen-lessons"}}` (`--dry-run` first: misses, new decisions, estimate). One
   native decision per miss (recall alias) gives three probabilities: reusable behaviour, already
   covered, supported by the recorded reply. The threshold is fitted on the fit split from saved
   same-tool deliveries and saved; unknown decisions are never redrawn. Screened-in misses become
   `unresolved` rule entries `derived_from` their trajectory; the rest stay source evidence.
3. `--audit N --include-inspected --audit-memory none`, then the same with `lessons` (only
   screened-in records and their overview lines, no raw observations): the same cases and specs
   except memory, reported as development re-audits of step 1's lineages.
4. Optionally `"resolution": "evolve-lessons"`, `lessons` set to the printed `lesson_set` and a
   fresh `run_id`: one proposed rule edit, replayed on held-out and protected fit lineages without
   memory, `accepted` only on a strict exact-reply gain without regressions; every attempt stays
   `research`. Pass the returned `library` to build on an accepted edit.

A held-out lineage stays independent only while nobody has read its per-trace results, or the
results of anything grouped with it. That fact is kept in one exposure ledger at the `.exp` root
(`<root>/exposure-ledger/ledger.sqlite`), shared by every project on the machine, so it survives
reimports, rebuilds and regrouping:

- Nodes are traces fingerprinted under a scheme, each pointing at its immutable verified source
  record (project, trace dataset, trace ID and the record's SHA-256) and keeping the source
  trace ID and any declared task identity (`task_lineage`). Project-local lineage IDs are not
  used, because they change on reimport and regrouping.
- Edges are grouping relationships that every `exp build` records, exposed or not: traces that
  declare the same task identity, and the build's own exact or semantic near-duplicate unions
  with their similarity. Each names the project and build that produced it. Nothing weaker (a
  shared customer, conversation or time bucket) is an edge.
- Equivalence links join one source's nodes under every scheme that fingerprinted it, for
  unexposed traces too.
- Exposures mark content a person saw, with the panel, receipt and report that showed it.
  `--verbose` shows per-trace detail, so it exposes every member trace of the audited lineages.
- Reservations claim a panel's content for one audit receipt before any of its runs exists.

The fingerprint (`trace-content` v1) hashes only the task instruction and the multiset of paired
exchanges (tool name, canonical arguments, paired recorded reply) over the whole recorded trace,
public prefix included, with duplicates counted. Calls and results are paired by
`gen_ai.tool.call.id`, never by position; a missing, repeated or colliding call ID leaves the
trace unresolved (no fingerprint), though grouping edges still reach it. IDs, timestamps, import
and exporter metadata, request-context keys (including `replay_from`) and call order never change
it, and content is kept verbatim. Ignoring order makes it conservative exposure matching: two
different execution sequences of the same exchanges match, which errs in the safe direction. A
scheme's identity is its name, version and the digest of its written specification; any change
to that specification needs a new version even if its golden vectors still pass. Golden vectors
must reproduce before a scheme computes or matches. Candidates are matched under every
computable scheme. A ledger recording the same name and version with a different definition is
refused, and so is such a scheme on `merge()`, whose nodes then stay as unknown-scheme nodes.

A held-out lineage is not independent when any member trace shares a connected component with
exposed content, shares one with content another receipt holds, or cannot be fingerprinted (the
whole group is ineligible, even if its selected trace is fingerprintable). Exposure is transitive
and ignores order: if exposed A was grouped with B and a later build groups C only with B, C is
exposed even after A disappears from imports, and edges recorded before an exposure still carry
it. Nodes under a scheme this machine cannot compute block every independent audit until an
equivalence resolves them (re-fingerprinting their source record when it is still available
and unchanged); the refusal reports their count and scheme, and development re-audits proceed.
Every table is append-only. Independence is only relative to the ledger a decision used;
`ExposureLedger.export()` and `merge()` share one between machines. A build made before the
ledger existed is refused until `exp build` is rerun on the same import to record its grouping.

Grouping links recorded after a reservation are reassessed whenever its receipt is resumed or
reports. If they reveal exposure that predates the receipt, an undispatched reservation is
released and the audit refuses to run independently; once execution began, an invalidation
notice is appended instead and the headline becomes "independent, invalidated: predating
exposure". The receipt and its snapshot are never rewritten, and exposure that
happens only after the receipt was minted never invalidates it.

An audit runs in a fixed order: reserve, register, claim a dispatch slot, call. Selection reads
the ledger, then the panel's exclusive lease (an OS file lock, `panel in use` for any second
process) is taken and, in one ledger transaction, the graph is read again and the reservation
committed; content another audit took meanwhile is never reserved. The lease is held through
execution, every slot claim first checks it, and locks are always taken in one order: panel
lease, ledger transaction, a run's execution lock, its dispatch ledger. Atomicity holds only for
runs sharing one ledger file on one host; SQLite and file locks over network filesystems are
unreliable.

Every audit has one immutable receipt (`held-out-audit-receipt`): the panel identity (a hash over
the sorted selected case IDs and each case's first-generation content-addressed run ID, which
covers the simulator, memory checkpoint and recall policy, every prompt sent and the judge
configuration), the code revision (not part of run IDs), the cases, the memory checkpoint (lineage,
version, scope, fit traces learned and simulator artifact), every fingerprint scheme's full
identity (name, version, definition and vector digests), the ledger snapshot it reserved against, and for each exposed or reserved
lineage its component size and edge path, so an over-long chain is visible. Its basis is
`independent` when no audited lineage was exposed or reserved by another receipt and no run had
a saved result, otherwise `development-re-audit`. Everything later is appended beside it: a
report per invocation (prepared, stopped, evaluation, or replay of fully saved results, which
keeps the receipt's headline) and a child per replaced registration. The same command resumes
its receipt when the cases, panel identity and code revision all match.

A run that claimed even one slot makes the reservation permanent, since its output may have
reached logs or screens; after a crash the receipt reports "dispatched, outcome unknown", and
only that receipt can complete it with `--retry-incomplete`, which resumes the same run in
place (below). A run that sent a lost request again becomes an in-place `re-attempt` child that
names each re-dispatched step by stage, and the headline becomes "independent, qualified: trace
re-attempts": the primary score keeps every trace, the report counts the re-dispatches by stage
(prediction, judge, investigation) and the affected traces, and it adds the score over
unaffected traces only and over re-attempted traces only, each with its own denominator. A registration retired before any dispatch (for example
after a catalog change) is a `re-registration` child and does not qualify the headline. When the
frozen configuration changes, an unused reservation is released by its owner under the lease
(after recounting every claimed slot) and a new receipt is minted; after any dispatch the old
reservation stays and the new receipt over those cases is a development re-audit.

By default an audit selects only unexposed, unreserved, resolved held-out lineages and says how
many it skipped and why. `--include-inspected` re-runs them too; the report then labels the panel
"Re-audit on inspected cases (development)" and never independent. When none remain, the
command refuses before any provider call: an independent audit needs fresh traces, a new import
with new content. The check runs when the command starts; if another audit reserves the last
eligible lineages while this one learns its fit-only memory, the refusal comes only after that
learning, which is saved and replays without calls on the next run. Every report states the ledger's limits: an
unmerged ledger on another machine cannot show its exposure, paraphrased or regenerated variants
that near-duplicate grouping misses are not caught, knowledge a person carries over by hand is
not tracked, atomicity is same-host only, and matching ignores call order.

The audit stays independent only while nobody tunes the simulator on its cases. Held-out traces
are selected deterministically, so a second audit replays the same ones. Once anyone inspects a
held-out trace's per-trace results (`--verbose`, the per-trace table or its transcript) and
changes prompts, rules or code in response, that trace is development data: a later score on it
is a re-audit of fitted cases, not an independent measurement. The provenance checks cannot see
this channel. An independent audit after such tuning needs held-out lineages nobody has
inspected, which in practice means new traces from a new import.

In every replay, turns the simulator never answered count as not reproduced, and `--verbose`
adds every trace and turn. Run IDs are derived from the run's inputs, including the prompts the
simulator sends, so a changed input starts a new run and an unchanged one replays without provider
calls. A run interrupted before it saved a result (a killed process, a lost connection) is never
bought again silently: the plain command stops on it, and `--retry-incomplete` (on
`exp build` or `exp eval --audit`) resumes it in place. The same run continues from its first uncommitted step, with the same run ID and the
memory it pinned when it started, in a fresh three-hour window. Committed steps, saved
investigations, a saved judge response and finished panel traces replay without provider calls;
if only the judge is missing, only the judge runs. A request that left the process without a
saved response has an unknown outcome: it may have been answered, billed and seen (in a gateway
log or a terminal). It is sent again and reported as `step k re-dispatched after an unknown
outcome` (k counts turns from 1) with its attempt count, so the retry is transparent but not
provably uninfluenced. A step whose interrupted preparation (query embeddings, memory-recall
decisions) must be prepared again is reported the same way, at stage `retrieval`. Such a trace is
a trace re-attempt: any learning, retest or audit report that includes one says
`qualified: trace re-attempts` in its headline, keeps every trace in the primary score, counts
the re-dispatches by stage and the affected traces, and adds the score over unaffected traces
only and over re-attempted traces only, each with its denominator; nothing is dropped by
default. A learning trace that saves a result but is
not learned (for example after a failed investigation) is retried up to three times in the same
pass; any trace still not learned is reported separately. A trace whose identical request the
gateway refuses deterministically (`invalid_provider_stream`) is retired after three refusals
under `--retry-incomplete`, and an operator can retire a stuck trace by name with the
`repair-learning` `retire-trace` resolution; a retired trace is counted as not learned, never
scored, and memory is unchanged ([reference](reference/learning-retirement.md)).
Every trace is predicted whole before any of its replies is disclosed;
investigations of its misses then run together, so a failed investigation never prevents a
prediction. A trace's first saved measurement fixes its score, including unanswered turns as
misses. Later learning retries cannot replace that measurement, and the curve reports traces not
learned separately. The retest covers learned traces only and preserves their first measurement
as its baseline. A cancelled learning preflight records nothing; later commands read the memory
that learned most recently. When every trace is already learned, learning replays its saved
results without asking for spend consent.

Knowledge pinned in a simulator (built outside `exp build`, for example for `exp eval --fidelity`)
is checked before any provider call. Its constructor sources, research notes and prior-library
entries are resolved to recorded traces and task lineages, and a task whose own evidence they
contain is refused. A held-out task is also refused when any knowledge channel has no recorded
provenance, unless the hand-written rules are declared `authored`.

A trace may mark its public prefix in its request context (`exp.request.context`, for example
`{"task_lineage": "user_17", "replay_from": 2}`): the first `replay_from` recorded tool calls and
their replies are given to the simulator as initial observations, and measurement starts at the
next call. Without it, every recorded call is measured. Traces that declare no tool list
(`exp.request.tools`) get tool signatures observed in learned traces: names and argument names
only.

A trace may also supply its instance facts, the starting world memory cannot know (workbook
cells, orders, tickets), under `instance_facts` in the same request context:
`{"facts": [{"key": "input.xlsx/Sheet1/A2", "value": "PQTW1", "visibility": "conditioning",
"provenance": {"adapter": "...", "source": "...", "validation": {"status": "validated",
"method": "..."}, "state_version": 0}}]}`. `exp build` (learning and its retest) and
`exp eval --audit` then deliver that trace's facts to its own predictions in a
separate `instance_facts` request channel, apart from memory and research history, and the
facts are part of the content-addressed run ID. The conditioning facts a learning pass delivers
also key its memory lineage, so adding, changing or removing them learns a fresh memory. Facts are moved out of the request context on
import, so they never reach retrieval grounding, mined tasks, recorded fingerprints or the
memory other traces read. `state_version` counts applied actions (0 is the starting state): a
prediction for the action at index i receives only facts of version i or earlier, so a fact read
after an action never conditions it. `verifier_private` facts reach no simulator or actor
request. A fact that is malformed or not `validated` stops the command with an error naming it.
A fact may list `derived_from_traces`; one derived from another held-out trace is refused, since
held-out-derived facts condition only that trace's own replay. Domain decoding (for example
reading a workbook) belongs in an environment adapter outside the core; in Python, an adapter
implementing `InstanceFactsProvider.initial_facts(task)` can also pass facts to
`replay_case(..., facts=adapter)`.

### Simulated tool observations

Text world models generate environment observations for declared tools. Candidate tool calls are
passed to the world model together with schemas, the visible transcript, retrieved trace examples,
and private environment state. Parallel calls receive one ordered tool message per call ID; no
external tool implementation runs. Malformed observation batches are invalid rollouts.

Durable learning and fidelity sessions validate state-edit structure and commit a transition
atomically. They do not mechanically enforce domain rules such as which ticket statuses may
follow one another, or whether a literal reply agrees with the proposed state. A structurally
valid wrong transition remains in the candidate world and can affect later predictions. Learned
rules are provisional model hypotheses; compression is lossy and cannot certify their truth.
Their full source records and provenance remain available. Structural correctness and measured
reply fidelity therefore do not establish equivalence to an executable application's behavior.

The public session API accepts `world.new_session(task="Research", tools=(tool_schema,))` followed
by `world.step(session.id, assistant_message)`. Each result exposes `messages`, an ordered tuple
of OpenAI user or tool messages, and `terminal`. Tool observations are nonterminal so the agent can
consume them before producing its final answer. World-model artifacts pin the current prompt; rebuild
projects created with a different prompt before running them.

### Materialize declared native outputs

`exp eval PROJECT --materialize SPEC.json --root .exp` materializes an explicit native edit plan
from an audited simulator checkpoint. Install the `artifact-edits` extra for the XLSX adapter.
The specification is a `NativeMaterializationSpec`: an exact `checkpoint` artifact reference,
a `NativeArtifactBinding`, and an optional new `destination` directory. `--dry-run` validates
frozen bindings without executing the adapter. It does not validate or execute individual edits.

Freeze `native_artifact_binding` in the task's initial context before simulation. The binding
names the immutable initial files, adapter definition digest, declared output paths and state key.
Supply the adapter's operation schema and initial native facts as public task context. The model
predicts the original environment's responses and separately records cumulative typed native
edits at the binding's state key. Capture the committed session with `capture_world_checkpoint`
or `capture_stored_world_checkpoint`, then pass its exact artifact reference to materialization.
A binding cannot be attached retroactively to an existing prediction.

The current CLI adapter applies ordered row deletions to initial XLSX bytes. It preserves native
cell values, types and number formats, and refuses formulas and other unsupported row dependencies.
It does not execute predicted Bash or Python commands. A native plan contains `input_sha256`,
`operations` and `expected_projection`; its operations start from the registered initial files.
The selected adapter executes these declarations and compares its actual projection with the
prediction. A mismatch persists a diagnostic receipt and refuses output delivery. It never repairs
the simulator state. Saved receipts reuse the same bytes without model calls or edit reexecution.

Harbor tasks with this binding expose `outputs --directory NEW_DIRECTORY` after submission and
declare those files for Harbor artifact collection. Such packages remain **diagnostic only**:
a matching prediction is not an independent task verifier, a solvability demonstration, or actor
tier calibration. Native verifier calibration is required before qualification. The XLSX adapter
is a bounded artifact transformer; general executable applications still need their own selected
`EnvironmentRuntime` and sandbox.

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
Set `exp eval PROJECT --models ALIAS,ALIAS --concurrency 512` to allow up to 512 rollouts
in total across the selected models. Python callers use
`ModelEvaluationOptions(maximum_concurrency=512)`. Pending work is interleaved across models;
the SDK admits only as many cells as the shared pool can execute. Provider rate limits and the
approved spending allowance still apply.
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

When a catalog model carries `gateway.prices`, preparation freezes the complete
`GatewayTokenPrices` in its request reservations and `CandidateTokenPrice.token_prices`.
Gateway metadata containing only capabilities leaves the model's existing flat rates in use.
An explicitly supplied empty price card instead declares unknown pricing and cannot be replaced
by those flat rates. Saving and reloading the catalog preserves this distinction.
Long-context thresholds apply to each request's original input total, never to the sum of a
rollout's calls. Cache reads and writes remain disjoint input subsets; reasoning remains an
output subset. Launch estimates also price each captured request before averaging source
episodes. Explicitly returned service tiers select their authored schedule. When no tier
is returned, valuation uses the ordinary request contract. Native gateway relays currently do
not preserve returned service-tier metadata, so their reports cannot establish a different tier.
`ModelRequest` cannot select flex or priority, so its estimates and finite admission checks use
the ordinary schedule and reachable long-context tier. The other authored cards remain frozen
for actual returned-tier valuation. Judging retries retain the original complete judge card.
These frozen attribution rates are separate from provider invoices, account discounts, and any
gateway debit. Current four-rate and full-schedule requests use the same wrapped response
contract. Receipts from a different response contract remain untouched and fail before dispatch;
create a fresh preparation instead of rebinding saved requests.

Unknown meters remain distinct from measured zero. Missing subset counts are priceable only
when every possible allocation has the same authored rate. A positive cache-write total with
no one-hour split is priceable only when both write rates are known and equal. No rate or meter
is inferred from a provider name. A missing applicable rate makes the quote's
`maximum_is_upper_bound` false; an aggregate cap cannot authorize that request. Uncapped
execution still saves paid responses and unknown liability before surfacing a valuation error.
Prepared evaluation with a finite cap rejects any incomplete stage before constructing provider
clients. Replaying a preparation with incomplete tariffs requires uncapped mode; saved receipts
remain untouched. Automatic and hosted router optimization require a finite allowance and reject
an incomplete candidate, world, or judge tariff during planning.
A priceable successful response cannot settle unbounded
liability from earlier potentially paid retries; certified unpaid attempts do not add liability.
Request-receipt replay returns the saved result or pricing error without another call, including
after the operator adds a lower cap. Online routed accounting retains an unknown total when
the final reply cannot account for earlier potentially paid attempts, preserving the original
response beside that total. Unpriceable assistant usage cannot create a known report cost.

## Objective-driven investigation reading

`exp build PROJECT --traces PATH --source SOURCE --root ROOT --evidence-reading` enables bounded
evidence reading during learning investigations; see `reference/evidence-reading.md`.

## Native and mixed-world continuations in Python

Pair chat context and remaining calls with a native/mixed checkpoint using
`ChatAgentRuntime.run(..., on_checkpoint=...)`, `ContinuationCheckpoint` and
`persist_continuation_checkpoint`. Capsules bind immutable sources, public tasks and binary files.

`exp.optimize.evaluation.continuations.study.run_continuation_study` consumes that capsule
through a `ContinuationStudySpec`, an exact matching actor/world runtime, a caller-admitted
model client, and a private state probe. The spec pins actor model, sampling, repeats, verifier and probe identity. Optional predicted success and selection
rationale are registered before outcomes. The caller must provide bounded provider admission;
this Python API does not authorize spend itself. The CLI below owns admission.

Every branch restores the same world and actor prefix without executing completed calls.
Goal and policy grades stay separate. Saved results replay; interrupted outcomes stay unknown. Only unstarted branches execute on another invocation. Reports retain all planned branches and disclose the assessable denominator for success rates and Brier scores.

Only fit tasks with checked trace provenance are accepted. Training needs independent, preregistered
acceptance evidence. Pi/process continuation and trained selectors/critics are absent. Native edits
refuse unsupported features or changed retained cell types/formats. Write replies acknowledge files;
`describe_workbook` pages exact sheet names/dimensions, and `inspect_workbook` reads typed cells.

### Dream consolidation and frozen execution

Add `--dream-every N` to a learning build for bounded consolidation after every N traces and the
last trace. Prediction and feedback precede consolidation and one memory-version publication.
Original observations get the first record slot; one prior wiki claim gets the next for revision, then remaining originals precede research context. Citations name only originals; page labels are optional. Checks include uncited counterexamples; hypotheses cannot support or veto claims.
When individual citations are inconclusive, one bounded joint read may check a multi-record claim.
Any local contradiction, malformed judgment or oversized reader request vetoes publication. Complete evidence is never truncated; support must cover every factual clause.
Plans pin the dream protocol; missing pins or changed prompts/schema/limits require a new project.
Accepted claims retain scope, uncertainty and provenance. Explicit transferability is required for overview rules; historical-only claims remain records and cannot supersede reusable rules. Rejects and retirements remain recorded.
Reader agreement is not native verification or demonstrated benefit. Lost outcomes never silently retry.
`RUN-memory-outcomes` joins full-claim delivery to later reply scores and pinned memory; summary-only
exposure is excluded. This observational ledger never promotes rules or enters learning; retention is labeled.

`exp.simulation.world_model.executable.GroundedWorldFactory` adapts the existing grounded
transition engine to the executable state machine. It retains predictor history with state,
uses compact edits (including authority-bounded `copy`), and cannot alter the kernel's clock or undeclared domains. Restoring a
checkpoint restores context and state exactly; subsequent model responses are still new draws.
Compose it with `MixedRealityEnvironmentRuntime` for native and simulated domains. Ownership is
explicit: a simulated edit cannot update a native workbook. Manifest `native_reads` bridges
map simulated arguments to private zero-duration native read tools. Reads run on isolated forks;
state, scheduler or RNG changes refuse. Bounded checkpoint/time-bound receipts condition only the
current prediction through `InstanceFacts`, never learned rules. Required probes are checked before due events and again before tool execution. Schema-invalid arguments and proven pre-dispatch access refusals return recoverable errors without advancing the world.
This grounds reads without synchronizing state. The predictor receives per-tool `reply_schema`;
shape violations leave state/history unchanged. Explicit `maximum_reasks` (0 to 3, default 0) permits bounded corrections of known refused proposals within the same physical budget; every response and refusal is retained, and successful observations report the refusal count. Lost responses and valid-but-wrong results are never retried by this mechanism. Domain `state_schema` applies at initialization, restore and commit. `reply_bindings` select current state using exact arguments, unique-member matches, exact ranges or explicitly clamped pagination. `workbook_contracts.simulated_workbook_manifest` binds these to full typed workbook projections. These checks establish internal consistency, not native fidelity.
The built-in `exp/optimize/evaluation/executable/grounded.py` factory's `GroundedRuntimeDefinition` pins the world-model reference, catalog,
optional exact `EvaluationMemory`, task, manifest, initial domain state, private projection,
deadline and cumulative physical-attempt ceiling. Its predictor, query embeddings and memory
readers share admission. Provider SDKs bypassing that transport are refused. Projections from
this factory have simulated authority; they do not establish native truth. Missing memory means
no learned memory is delivered. Optional `facts` (`ReplayFacts`: the same validated, provenance-bearing
`InstanceFacts` and held-out reservation rules as fidelity replay, with `first_version` the start's state
version) condition every simulated prediction in the separate `instance_facts` channel with the same
guidance, beside native read receipts. Facts are not state: they are never a writable domain or part of
a checkpoint image, and nothing is derived from native state. Only starting-state facts are accepted, and
the `native-read/` key prefix is reserved. Facts re-key the definition, ledger and backend digest, so a
checkpoint restores only under the same facts; definitions without facts keep their identities. Declare
them under `simulated` in the mixed definition, or in any continuation or `compare-environments` job. `workbook_runtime.py` binds immutable XLSX inputs and private worksheet
digest views. `mixed_runtime.py` combines a native factory with this simulator and real/simulated routes under one
manifest; private projections stay under separate `native` and `simulated` keys. Execution receipts
bind release plus installed EXP source/resource digest; code changes require fresh qualification. Third-party dependencies remain host-owned. Owned executable-world shadow routes refuse at setup until an independent mirror contract exists; cross-owner writes remain unsupported.

### Continuation, Harbor and training CLI

`exp eval PROJECT --continuations SPEC.json` runs one action from a JSON spec: continuation
studies and campaigns, witnessed SFT export, trajectory trace export, learning repair (lesson
screening and library evolution), saved-evidence analysis, environment comparison, task-factory
authoring and executable Harbor tasks. Every action, its payload and the Harbor reward contract
are documented in [docs/reference/continuations-and-harbor.md](reference/continuations-and-harbor.md).

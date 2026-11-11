# Release scope

This release supports the current source and one wheel with either core dependencies or the
optional `sft` dependency extra on their documented local paths. It claims only behavior exercised
on the exact release checkout.

## Supported and verified

- Root CLI commands are exactly `build`, `capture`, `config`, `eval`, `login`, `optimize`, and `run`; an invocation with no subcommand
  opens the default gateway home screen. Optimizer commands are exactly `router` and `model`.
- `exp login` opens the Platform approval flow for Experiential Cloud, stores the returned
  organization key in the user-data credential file, and synchronizes the authenticated account's
  hosted provider/model identities into the secret-free project catalog; no credential value is
  written to the project.
- `exp capture` is an experimental foreground macOS HTTPS collector on Python 3.13+ with normal
  Platform login, streamed OpenAI/Anthropic protocol capture, asynchronous trace uploads, and
  bounded private retry files. It uses pinned Experiential mitmproxy forks and defaults to supported
  provider hosts across all apps, with an optional exact-host override. The signed Mitmproxy
  Redirector app and its approved Network Extension own interception. Capture does not edit
  hosts, DNS, or system proxy settings and has no reset subcommand. Each selected host set gets
  a CA with critical certificate-level name constraints and persistent current-user SSL trust.
  Explicit client certificate rejection switches the affected process and
  host to encrypted pass-through for the run; missing process identity or exhausted capacity
  uses an explicit host-wide exclusion. Other capture continues, and the terminal names excluded
  targets and retains a partial-capture indicator. Backend failure still stops Capture. A quiet
  DNS guard checks selected providers before startup, stops after repeated lookup failures, and
  checks recovery after shutdown. The native selector excludes the macOS DNS responder process,
  but app-attributed DNS can still be intercepted. macOS UDP sockets follow application closure
  rather than idle expiry, and native EOF releases their forwarding tasks. A separate watchdog
  is armed before interception and disables it when the owner exits or its heartbeat expires.
  Live Codex capture, idle DNS reuse, owner crash, frozen-owner recovery, restart, and Ctrl+C
  shutdown have been exercised. These tests do not establish universal client compatibility or
  guaranteed network recovery. Fork distributions must be published before a package release;
  development checkouts use immutable source pins.
- The local gateway supports explicit provider references, identities, virtual keys, grants,
  singleton and certified ordered exact-model pools, frozen-project aliases, bounded precommit
  provider fallback, Chat Completions, Responses (over HTTP and as the Responses-over-WebSocket
  transport on the same route), bounded in-memory continuation and replay,
  content-free SQLite accounting, monthly integer nano-USD enforcement, loopback-only health
  and usage views, and optional identity-scoped guardrails that stay off until a policy is
  assigned. The bundled standard classifier pack is also default-off and is enabled only
  when one organization and identity opts in and binds an adapter for every capability.
- Native `POST /v1/systemone` serves TypeSafe decision requests with `noul`, `choice`, and
  `score` questions and typed answers, independently of conversational APIs. Admission requires
  a direct exact-model pool, explicit `gateway.capabilities.supports_decisions`, a known
  nonnegative input token rate, and output token rate zero. Requests are bounded to 32 questions,
  64 choice or 10 score criteria, and 262,144 bytes. Accounting reserves bounded per-question
  estimates, not provider-enforced output limits, and settles only provider-reported usage.
- Native `POST /v1/decisions` serves OpenAI Decisions API requests (`predicate`, `choice`, and
  `score` questions over text or inline base64 images) on the same decisions surface, through
  direct OpenAI deployments that declare decisions support and price output at zero, billed on
  reported input tokens. A Gemini deployment serving `gemini-decision-1` serves the same request
  shape through Google's Decisions API: the body is translated, the answer translated back, and
  what Google cannot express (non-key choice values) is a typed 400 before dispatch.
  Real Rust HTTP and SQLite tests cover all answer types, authentication, unsupported inputs,
  missing or invalid usage, certified 401 fallback, cancellation, timeout, and content-free accounting.
- The no-subcommand default gateway launch, the direct `exp run [PROJECT]` form, and the
  `exp --project PROJECT [--ghost]` compatibility form are installed-wheel surfaces. Gateway
  startup is provider-idle and requires explicit authority.
- The installed-wheel gateway lane uses real SQLite, a real subprocess listener, a real loopback
  upstream, and OpenAI `3.0.0`. It covers `OpenAI` and `AsyncOpenAI` across Chat Completions and
  Responses, with both stream and non-stream requests. HTML and JSON usage are checked for the same
  per-identity accounting values. The same lane measures default SDK retries against physical
  attempts, provider-authentication fallback, refusal policy, post-commit no-switch, restart and
  replay behavior, revocation, cancellation, WAL mode, and mixed `host_managed`/
  `customer_managed` cost attribution that remains frozen across restart and catalog replacement.
- Real-SQLite monthly budget evidence covers atomic concurrent reservations, provider-only route
  exhaustion, shared identity quota errors, billable failure and crash retention, retry and
  fallback accounting, keyed replay without duplicate spend, explicit schema migration, and UTC
  month rollover without a reset job. The real loopback waterfall is configured through the
  interactive-capable CLI and returns OpenAI `insufficient_quota` after shared exhaustion.
- Deterministic source-level certification additionally covers selection-only project routing,
  authentication-circuit open/skip/recovery, concurrent multi-identity key revocation and alias
  revision activation, and atomic rollback of a failed legacy SQLite migration.
- One content and secret canary scanner covers the gateway database, live WAL, migration backups
  when present, catalog snapshots, stdout, stderr, logs, usage responses, and HTTP error bodies.
- Public Python exposes provider-free build, explicit router composition, frozen selection-only
  router load through the normal gateway application, structural text-versus-sandbox comparison,
  and managed SFT composition. No separate router HTTP or SSE implementation is shipped.
- W16 router evidence uses 100 normalized traces, 50 fit tasks, 20 held-out tasks, 140 planned
  cells, 130 deterministic text simulations, and 140 deterministic judgments under one finite
  simulation and judgment budget. Observed hosted-service spend is exactly $0.00.
- W16 sandbox evidence compares two exact post-lock text and local-process pairs on a host that
  provides local process containment (Darwin, or Linux with unprivileged namespaces). It retains
  one malformed sandbox failure in the denominator and claims structural terminal agreement only.
- Exact-checkout CI supplies the full 40-hex Git revision, recursively verifies every evidence
  artifact and manifest input, and publishes machine-readable JSON plus JUnit evidence.

## Gateway provider evidence matrix

This table separates deterministic protocol evidence from hosted calls that need account
credentials. `Not run` means exactly that; it is not inferred from fixture coverage.

| Provider surface | Deterministic evidence in this release | Credential-gated live evidence |
|---|---|---|
| OpenAI | Native Responses fixtures for text, tool arguments, usage, cancellation, and refusal; all eight official SDK quadrants run against the installed local gateway | Not run; requires an OpenAI credential |
| Anthropic | Native Messages fixtures for text, tool arguments, usage, cancellation, and refusal through both public gateway surfaces | Not run; requires an Anthropic credential |
| Generic OpenAI-compatible | Real loopback upstream through the installed gateway; text, tool arguments, usage, cancellation, and refusal contracts | Not run; requires a compatible hosted endpoint and credential |
| Azure OpenAI | Compatible-adapter fixtures for text, tool arguments, usage, cancellation, and refusal | Not run; requires Azure endpoint and credential |
| OpenRouter | Compatible-adapter fixtures for text, tool arguments, usage, cancellation, and refusal | Not run; requires OpenRouter credential |
| Gemini | Native fixtures for text, structured complete function arguments, usage, cancellation, and refusal | Not run; requires Gemini credential |
| Amazon Bedrock | Native EventStream fixtures for text, incremental tool arguments, usage, bounded cancellation, refusal, and single dispatch | Not run; requires an authorized AWS account and region |
| TypeSafe SystemOne | Real Rust HTTP listener and SQLite with a synthetic loopback upstream; three typed answer forms, exact token settlement, error validation, certified 401 authentication fallback, cancellation, and timeout | Direct TypeSafe API smoke with synthetic input succeeded for all three question types on 2026-09-16; no hosted-gateway or deployed-fleet verification |

The machine-readable dated conversational matrix is
`exp/runtime/gateway/provider_certification.py`. Its live cells remain
`not_run_requires_credentials`; the separate TypeSafe smoke above does not certify those cells.
Gemini complete structured function arguments are explicitly not labeled as provider-byte
incremental tool-argument streaming. A direct TypeSafe success verifies that request and its
reported usage, not account limits, price-invoice agreement, production availability, or latency.

## Explicitly excluded

- System-wide Capture has not been certified against live Codex or Claude Code sessions. Their
  process trust stores and existing connections may require configuration or a restart. The
  signed macOS Network Extension requires user approval; synthetic tests do not establish that
  a real machine has approved it. The feature does not bypass certificate pinning or claim
  complete capture of unselected domains or unsupported protocols.

- There is no budgets dashboard. Monthly allocation management and remaining-allocation reporting
  are explicit interactive or non-interactive CLI operations.

- No paid E2B or Harbor cloud smoke ran. The repository verifies the optional `bounded-close-v1`
  Harbor lifecycle and ledger with injected fakes, but makes no cloud cleanup, provider-quality, or
  environment-parity claim.
- No real Tinker training ran. Managed SFT remains fail-closed unless its immutable configuration
  has a finite positive ceiling and its backend supplies a conservative full-schedule estimate.
- No trained-versus-base behavioral comparison ran because this release produced no paid training
  artifact. It makes no trained-model quality-improvement claim.
- The deterministic W16 evidence used no hosted model, judge, embedding, telemetry, environment,
  credential, or `.env` path and reports exactly $0.00 observed service spend. The separately
  authorized TypeSafe direct API smoke is not part of that provider-free evidence.
- Deterministic gateway certification uses a real loopback upstream and local SQLite. The
  conversational provider matrix's live cells did not run. Neither these fixtures nor the one
  direct TypeSafe smoke establishes hosted-gateway availability, account limits, or invoice accuracy.
- SystemOne has no chat or Responses conversion, streaming, tools, generation controls, project
  selection, chat guardrail processing, continuation, or idempotency replay. An inbound
  `Idempotency-Key` is ignored, so a repeat submission is a new request. Execution is capped at
  eight certified deployments and one dispatch each, with no same-rung or throttle redials;
  HTTP 402/429/529, ambiguous transport outcomes, and malformed answers are terminal unknown outcomes
  that keep their monetary hold rather than automatically retrying or failing over. Only HTTP
  400/401/403/404/422 establish a known rejection eligible to release that hold.

These exclusions are product boundaries, not evidence that the corresponding hosted services are
unsafe or unsupported forever. Any future claim requires separately authorized, finite-budget,
denominator-preserving evidence.

## Development task factory

Factory packaging requires task-quality assessment in addition to observable-contract
qualification. By default a separate LLM rubric proxy uses the last pinned panel model; set
`CandidateBatchSpec.quality_proxy` or `FactoryFamily.quality_proxy` to override its frozen model,
rubrics, deadline and physical-attempt allowance. It judges all six protected axes with cited
findings: prompt/verifier alignment, verifier/intent alignment, production prompt distribution,
world sufficiency and causality, failure attribution, and meaningful difficulty. All axes must
pass to package. Repairable defects or insufficient evidence remain `needs_review`, meaning
revision or another explicitly registered experiment, not mandatory human approval. A drop is
`rejected_quality`. An optional operator-attested human decision can replace the proxy.

Proxy requests contain public task/intent, the capability brief, the verifier, saved execution
outcomes and development controls; confirmation labels are withheld. The complete input and
output schema, model snapshot, rubric revisions, response and dispatch receipts are retained.
The host derives acceptance; the model cannot set an approval flag or claim label authority.
Catalog/attempt/deadline/capacity checks and shared spend consent apply. Missing or uncertain
responses are not silently redrawn; completed judgments replay without new provider calls.
Proxy approval is provisional model judgment, not independently calibrated expert agreement.
It permits automatic factory operation without requiring human review. Direct mechanical
qualification remains explicitly narrower.

This review is the semantic gate of the packaging path (`evaluate-candidates` and the
`--pipeline` factory). Task validity (`check-task-validity`: re-derived controls, literal and
judged answer leakage, closed-book draws) and hardening (`harden-tasks`) remain the mechanical
screens of scenario-search variants, documented in
[task hardening](reference/task-hardening.md). A proxy pass does not replace a validity
rejection, and hardening keeps variants without packaging them.

World history can be materialized through `RecordScenarioSpec.setup_actions` and `finish_tick`:
the existing kernel executes setup actions and scheduled events, replays native setup, and saves
the reached checkpoint and execution receipts. Causal consistency follows only from the declared
transition rules and checked invariants. Randomness must be seeded and its distributions justified;
a consistent synthetic history does not independently establish realistic production behavior.

`exp eval PROJECT --root ROOT --pipeline pipeline.json --dry-run` validates a
`exp.optimize.evaluation.factory.runner.TaskFactorySpec`. Repeat with `--yes --non-interactive`
to authorize the displayed finite author/actor allowance when prices are unknown. The manifest
binds seed tasks, capability briefs, independent control witnesses, the shared actor panel,
coverage strata, deadline and export destination. Runtime inference keeps its separately declared
allowance. Consent sets one spend ceiling over every author and actor request, the known-rate
subtotal. Unpriced roles may use only their displayed physical attempts, and only after an
explicit confirmation.

Each seed uses the existing two-to-four-candidate author, native feasibility, observable-contract
qualification, actor panel and Harbor exporter. Related source lineages, traces or equal workflow
signatures form one connected component. Variants and actor attempts do not increase family
counts. Coverage targets remain fixed, and failed or rejected candidates stay in the inventory.
Impossible planned coverage dispatches no measurements. Saved results verify exported bytes;
publication recovery reuses saved child results without purchasing them again.

This is development qualification over supplied seed worlds and witnessed controls. It does not
discover arbitrary future capabilities, generate a complete application, provide protected release
confirmation, prove native reconstruction accuracy, calibrate subjective judgments, demonstrate
training benefit or certify container isolation. Those claims need their own evidence. No weights
are trained. The factory does not promote its reports into independent label authority.

`TraceCouncilSpec.actor_harness` selects `chat` or `pi` for every actor in a council. Pi requires an
absolute executable path, its `executable_sha256`, a declared version and optional dependency-file
digests. New execution checks those pins; completed reports replay without the installation.
Both adapters receive public task context and instruction roles. Pi calls and output tokens are
bounded by the council policy; tools still execute through its injected simulator. Declared file
pins do not establish complete dependency closure or filesystem/network confinement.

Executable predictors receive only the current tool's declared read/write domains. The manifest's
write roots constrain both the advertised proposal schema and local edit admission; the kernel
owns clock advancement. Forbidden edits are refused with the raw proposal retained, never silently
removed. This contract requires the typed `json-tool` proposal protocol.

Optional `ToolEffectPolicy.reply_bindings` compare decoded reply fields with a backend's private
before/after state projection before commit. Missing fields refuse the transition. These bindings
establish declared local relationships only: they do not establish native reconstruction fidelity,
complete domain semantics or correctness of an independently authored goal. SQLite exposes table
contents; artifact worlds expose file existence, hashes and sizes; simulated worlds expose their
candidate domains. Unsupported projection requirements fail before backend dispatch.

Execution receipts distinguish calls not started, rolled-back tool effects, committed effects and
unknown effects. They bind checkpoint digests and retain scheduled-event changes separately from
tool rollback. Record business errors discard staged events and random draws with their transaction.
Continuation actors can recover from an unchanged, pre-dispatch access refusal while the attempted
operation remains in the evidence. Unknown effects never authorize automatic retries. Mixed routers
retain authoritative effect receipts; a stopped mixed session still requires explicit recovery.
Known failed executions remain incomplete and ineligible for training; their proven rollbacks are
reported separately from ambiguous outcomes. Harbor trials retain this distinction after restart.

`ActorComparison` requires separate reference and candidate execution assessments. Task checks move
with their trace during A/B counterbalancing. Incomplete or failed execution slots are unassessable
without judge dispatch. Equivalent observed actor mistakes may support fidelity, but never establish
actor competence. Rubric normalization remains distinct from mechanical accuracy.

`exp eval PROJECT --root ROOT --continuations SPEC.json` accepts
`{"action":"compare-environments","equivalence":...}`. Each `EquivalenceSpec` names a frozen
trajectory-derived judge rubric (`rubric`), and each case binds a reference and candidate
`ContinuationJob`, identical public actor checkpoints and policies, and the declared
initial-state and material final-state probe paths (both required under a full-state rubric).
The rubric comes from admitted development trajectories of other families through
`derive-equivalence-rubric` and an explicit `freeze-equivalence-rubric` author review; it is
provisional, not an independently calibrated judge. The candidate may use the
frozen grounded simulator or mixed runtime. Separate reference, candidate and repeated-reference
executions retain every planned branch, including unavailable ones. The reference repeat exposes
actor variability; it does not establish a confidence interval or population equivalence.
`--dry-run` checks the frozen contract without calls, and completed results replay without loading
runtime code or credentials. New dispatch needs the normal spend consent.

The report separates mechanical outcome/effect agreement, task competence, unavailable-pair
bounds and semantic rubric ratings. Missing initial evidence, changed initial actor requests or
incomplete execution make the comparison unassessable. Missing final-state evidence leaves the
material outcome unknown while the paired behavior is still judged. Matching failures do not become successful
training examples. No judge score overrides mechanical disagreement. Private evaluator probes
and council findings never enter the actor requests.
Blinded A/B judge packets exclude backend observation metadata and execution-owner receipts;
the council retains that provenance separately for diagnosis. Task content may still reveal its
environment, so this is removal of explicit backend tags, not proof of complete blinding.

The paired trace council inspects actual actions, replies, state projections, delivered message
fragments and failure receipts for each actor/task case. Findings require citations into that
paired evidence and remain diagnostic proposals. Invalid reviews and unknown dispatch outcomes
are retained without redrawing; saved reports must reproduce from original execution and review
artifacts. This path resumes the built-in Chat actor only. The separate simulator-only council
supports Pi, but paired Pi checkpoint execution is not implemented. Neither path establishes
independent analyst calibration, protected held-out status or training admission.

Continuation SFT export binds each complete response batch to its ordered saved tool results.
Every call must have an explicitly successful observation; error-producing batches remain intact
as context for later recovery and are excluded as targets. Tool observations in later model
requests must match those saved results. Passing final-workflow checks does not approve earlier
tool errors. These checks establish observed execution within the declared world, not each
action's usefulness or native/simulator actor equivalence. Free-text explanations still require
separate judgment authority and remain excluded from targets on this mechanical export path.

`search-scenarios` under `exp eval --continuations` accepts `ScenarioSearchSpec`: two to sixteen
prospective native setup definitions, independent acceptance references, and proposed solution,
alternative, recovery, targeted-error and unwanted-effect scripts. It executes existing setup and
witness owners, retains every candidate and checks discrimination before publishing a related
`scenario_variation`. A recovery control requires an observed error before completion; a targeted
error must fail the goal through permitted effects. Supplied scripts and labels cannot approve
themselves. Construction definitions and controls must currently be supplied; this is not automatic
discovery of useful capabilities or arbitrary executable applications. New variants retain their
parent lineage and clear inherited critic, selection and training evidence. They still require fresh
qualification before qualified Harbor delivery. No hosted-model accuracy or container-containment
claim follows from provider-scripted conformance tests.

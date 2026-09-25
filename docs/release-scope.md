# Release scope

This release supports the current source and one wheel with either core dependencies or the
optional `sft` dependency extra on their documented local paths. It claims only behavior exercised
on the exact release checkout.

## Supported and verified

- Root CLI commands are exactly `build`, `capture`, `config`, `login`, `optimize`, and `run`; an invocation with no subcommand
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
- W16 sandbox evidence compares two exact post-lock text and Darwin local-process pairs. It retains
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

# Release scope

This release supports the current source and one wheel with either core dependencies or the
optional `sft` dependency extra on their documented local paths. It claims only behavior exercised
on the exact release checkout.

## Supported and verified

- Root CLI commands are exactly `build`, `config`, `optimize`, and `run`; optimizer commands are
  exactly `router` and `model`.
- The local gateway supports explicit provider references, identities, virtual keys, grants,
  singleton and certified ordered exact-model pools, frozen-project aliases, bounded precommit
  provider fallback, Chat Completions, Responses, bounded in-memory continuation and replay,
  content-free SQLite accounting, monthly integer micro-USD enforcement, and loopback-only health
  and usage views.
- Both no-argument gateway launch and the retained `exp run PROJECT [--ghost]` compatibility form
  are installed-wheel surfaces. Gateway startup is provider-idle and requires explicit authority.
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

The machine-readable dated matrix is
`exp/runtime/gateway/provider_certification.py`. Its live cells are
`not_run_requires_credentials`. Gemini complete structured function arguments are explicitly not
labeled as provider-byte incremental tool-argument streaming.

## Explicitly excluded

- There is no budgets dashboard. Monthly allocation management and remaining-allocation reporting
  are explicit interactive or non-interactive CLI operations.

- No paid E2B or Harbor cloud smoke ran. The repository verifies the optional `bounded-close-v1`
  Harbor lifecycle and ledger with injected fakes, but makes no cloud cleanup, provider-quality, or
  environment-parity claim.
- No real Tinker training ran. Managed SFT remains fail-closed unless its immutable configuration
  has a finite positive ceiling and its backend supplies a conservative full-schedule estimate.
- No trained-versus-base behavioral comparison ran because this release produced no paid training
  artifact. It makes no trained-model quality-improvement claim.
- No hosted model, judge, embedding, telemetry, environment, credential, or `.env` path was used by
  release evidence. The deterministic W16 evidence reports exactly $0.00 observed service spend.
- Deterministic gateway certification uses a real loopback upstream and local SQLite. No live
  provider matrix cell ran, so hosted availability, account limits, billing, and service-specific
  behavior are not claimed.

These exclusions are product boundaries, not evidence that the corresponding hosted services are
unsafe or unsupported forever. Any future claim requires separately authorized, finite-budget,
denominator-preserving evidence.

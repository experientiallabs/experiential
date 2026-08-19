# Local gateway architecture

## Supported surface

No-argument `wmo run` starts an authenticated multi-alias gateway on `127.0.0.1`.
It serves:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /health/live` and `GET /health/ready`
- `GET /usage` and `GET /usage.json`

The separate `wmo run PROJECT` form remains the frozen single-project compatibility server.
Gateway startup and readiness perform no provider request. Only an authorized model request may
cross the provider boundary.

## Authority and management

`wmo config gateway` owns explicit local setup. Its provider, identity, key, grant, alias, and pool
commands produce versioned receipts suitable for interactive or non-interactive callers. There are
no runtime seeds. A usable installation requires an organization, active identity, active virtual
key, explicit identity-to-alias grant, active alias revision, immutable catalog snapshot, and a
resolvable provider credential reference.

Private authority state lives in `ROOT/gateway/gateway.db`. SQLite uses WAL mode, versioned forward
migrations, private backups before migration, newer-schema refusal, and serialized initialization.
Virtual keys are stored only as peppered fingerprints. Key material is delivered once in a JSON
receipt or to a new mode-`0600` file, and commit ambiguity preserves recoverability. Provider
configuration stores an environment variable name, never its value. The local pepper is mode
`0600` and is not exported.

Every data-plane request is authenticated and authorized before request decoding, routing,
continuation lookup, or provider work. Authorization freezes organization, identity, API surface,
alias revision, target, catalog digest, request digest, optional hashed operation identity, and one
monotonic deadline. Identity disable, key revocation or expiry, grant removal, and alias revision
changes fail closed.

## Catalog, aliases, and exact-model pools

`wmo/common/models/catalog.py` is the authored provider and deployment catalog. Gateway snapshots
under `ROOT/gateway/catalog-snapshots/` are immutable, secret-free views of an exact catalog digest.
An advertised alias is executable only when readiness holds for the exact tuple of alias name,
alias revision, and catalog digest used by authorization.

An alias targets either:

1. a direct exact-model pool, or
2. one immutable project activation.

A singleton alias creates a one-deployment pool. `wmo config gateway pool certify` can replace that
with an ordered pool only when every member has the same exact logical model identity and an
operator-supplied equivalence certification. The certification records an ID, provenance, evidence
digest, time, and exact deployment order. Project policy selection still chooses one exact logical
model; operational fallback can only move among certified deployments for that model.

## Request, route, and provider attempts

The content-free ledger accepts the logical request before learned project selection. Selection or
direct resolution then produces an execution snapshot containing the exact model, pool, and ordered
deployment IDs. Each physical provider dispatch gets its own durable attempt row immediately before
network work. Attempt ordinal counts all physical dispatches; route depth identifies the selected
deployment position.

Provider execution is always internally streaming. Bounded same-deployment retries and ordered
deployment fallback are allowed only for typed precommit failures. The first outward text, refusal,
or tool-call semantic event commits the deployment, after which the gateway never switches
providers. Typed refusal fallback is disabled unless the active alias revision explicitly enables
it. Opted-in refusal deltas are withheld only in a bounded in-memory buffer: a refusal-only terminal
result can advance to the next certified deployment, while mixed semantic output or buffer overflow
commits and flushes the original route. Provider-internal retry layers are disabled so every
possible billable dispatch is visible to the gateway ledger.

Each physical attempt records its own provider, model, usage, latency, terminal state, and estimated
cost attribution. The parent request terminalizes once after success, final failure, cancellation,
disconnect, or crash reconciliation. Unknown prices remain unknown instead of being treated as
zero or copied across deployments.

## OpenAI-compatible protocol

Chat Completions and Responses have separate allowlist decoders and field-specific OpenAI error
responses. Both convert to one canonical gateway request without conflating their wire contracts.
Chat streaming emits valid completion chunks and one `[DONE]`. Responses streaming emits the
created, in-progress, output, and exactly one terminal lifecycle. Provider tool-argument fragments
are accumulated in original order and validated only at the complete-call boundary.

Commit-independent headers are available before streaming begins. Route-dependent headers are
emitted only after an execution snapshot exists. Stable public IDs do not expose raw key,
idempotency, request, or provider values.

OpenAI `3.0.0` `OpenAI` and `AsyncOpenAI` clients are release-certified for Chat Completions and
Responses in synchronous and asynchronous, streaming and non-streaming forms. Responses
continuation and duplicate replay retain content only in bounded, process-local, tenant and
alias-revision-scoped stores. Replay is opt-in through an idempotency or client request key. Restart
or eviction returns an explicit unavailable error and never reconstructs content from SQLite.

## Content-free observability and lifecycle

SQLite stores hashes, frozen authority, route identity, state transitions, token counts, latency,
and estimated cost. It never stores prompts, responses, raw tool arguments, raw virtual keys, or
provider secrets. `GET /usage` and `GET /usage.json` are two renderings of the same versioned report
and expose only aggregate and per-identity accounting. Estimated cost is attribution, not a provider
invoice.

The process owns readiness from preflight through bounded drain. New work is rejected after drain
starts. Admitted tasks, upstream streams, disconnect cleanup, replay ownership, continuation state,
and final ledger settlement are process-owned and bounded. A stuck cancellation cannot prevent the
terminal flusher from attempting content-free settlement.

## Certification boundary

Deterministic release evidence uses a built and freshly installed wheel, real SQLite, a real
subprocess-bound loopback gateway, a real loopback upstream, and the official SDK clients. One
scanner checks database, WAL, backups when present, catalog snapshots, stdout, stderr, logs, usage
responses, and error bodies for raw content and secret canaries.

`wmo/runtime/gateway/provider_certification.py` is the dated provider capability matrix. Each cell
names the official client SDK, public gateway surfaces, provider wire surface, fixture result, and
credential-gated live status. OpenAI and Anthropic have native fixtures; generic OpenAI-compatible,
Azure, and OpenRouter share compatible-stream coverage; Gemini and Bedrock have native deterministic
fixtures. Live provider cells remain explicitly `not_run_requires_credentials` until a separately
authorized run supplies dated evidence. Deterministic fixtures do not imply hosted-provider
availability, billing, or account-specific behavior.

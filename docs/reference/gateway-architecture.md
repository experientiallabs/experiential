# Local gateway architecture

## Supported surface

`exp` opens the gateway home screen. Its `Run Gateway` choice starts an authenticated
multi-alias gateway on `127.0.0.1`.
It serves:

- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses` as a WebSocket upgrade (the Responses-over-WebSocket transport used by
  the Codex CLI against api.openai.com: `response.create` request frames, one standard
  Responses stream event JSON per text frame, wrapped `{"type": "error", ...}` frames for
  request-level failures, and a `generate: false` prewarm answered without provider work; the
  bearer key is authenticated before the upgrade is accepted, and a GET without a well-formed
  upgrade answers 426, the status the Codex client maps to its HTTP fallback)
- `POST /v1/messages` (the Anthropic Messages API; `POST /v1/messages/count_tokens` answers an
  explicit Anthropic-shaped refusal because the gateway has no tokenizer authority)
- `POST /v1/embeddings` (the OpenAI Embeddings API: message-less and never streamed; served
  only by aliases whose catalog capabilities declare `supports_embeddings` on an OpenAI-wire
  connection, billed on the provider's reported `prompt_tokens` with no output leg, and
  returned with the provider's exact vectors in `float` or `base64` form; an inbound
  `Idempotency-Key` is ignored because the surface has no replay protocol)
- `POST /v1/images/generations` (the OpenAI Images API, generations only: prompt in, images
  out, never streamed; served only by aliases whose catalog capabilities declare
  `supports_image_generation` on an OpenAI-wire connection, billed on the provider's reported
  prompt and image tokens, so a model that answers without token usage is refused as
  unbillable rather than served for free)
- `GET /health/live` and `GET /health/ready`
- `GET /usage` and `GET /usage.json`

`exp --project PROJECT` is compatibility sugar that activates one project-backed alias and launches
this same gateway application. It does not create a router HTTP server. Gateway startup and readiness
perform no provider request. Only an authorized model request may cross the provider boundary.

## The data plane

The gateway has exactly one data plane: a native Rust HTTP server compiled as
a PyO3 extension (`exp_gateway_native`). Every launch path serves through it,
and a missing compiled extension fails the launch with the exact build
command rather than falling back.

The native engine owns the public socket and every serving fast path:
upstream dispatch, provider stream normalization, the certified deployment
waterfall, and public SSE encoding run off the GIL, with JSON-string
callbacks into python per request (authenticate, admit, `start_attempt` per
physical dispatch, settle, and `enforce_output` only when admission sets
`output_guardrail`). Unguarded traffic never calls that output callback.
Everything protocol- and authority-shaped stays in python: admission decodes
the raw body with `decode_chat`, enforces the deployment-identity invariant,
builds every deployment's upstream payload with the `streaming_requests`
builders, and writes durable SQLite transactions over hot-reloadable
authority generations.
Provider wire facts come from the public `gateway_wire_profile()` on each
resolved provider client; native dialects are `openai_responses`,
`anthropic_messages`, `openai_compatible` (which also covers Azure and
OpenRouter connections), `gemini_generate_content`, and
`bedrock_converse_stream`, so every granted provider has a native dialect.
Bedrock streams the AWS binary event-stream framing rather than SSE, and it
authenticates with per-request SigV4 signatures: admission freezes the exact
serialized Converse body, and the data plane signs it python-side through the
`sign_dispatch` callback (credentials never cross the boundary) after its
bounded dispatch permit and immediately before the provider POST, then sends
the frozen bytes verbatim. Signing at dispatch time means queue wait can
never age a signature toward AWS's short clock window; the engine's immediate
bounded open retry reuses the result within milliseconds, and any later retry
is a fresh admission and a fresh signature.

Multi-deployment certified pools execute natively. Admission returns the full
ordered route plus the frozen retry-policy facts without starting an attempt;
the engine reserves each physical dispatch through `start_attempt`
immediately before network work, redials the same deployment only for
retryable failure classes, fails over to the next certified deployment for
failover-eligible failures, and permanently freezes the serving deployment at
the first outward semantic event. Candidate selection policy (health
circuits, budgets, attempt caps) stays in python. When the alias revision
enables refusal failover, refusal deltas are withheld in a bounded in-memory
buffer so a refusal-only terminal can advance to the next deployment; mixed
output or buffer overflow commits and flushes.

Unknown routes answer a native 404 in the OpenAI error envelope, keyed Chat
Completions and keyed Responses run the replay protocol natively (the
Messages surface defines no idempotency header and never joins either replay
store), and `/usage` plus `/usage.json` are served natively. Startup
validates that every granted alias is natively servable (every pool
deployment resolves to a provider client with a native dialect) and fails
with the offending aliases named otherwise. Shutdown drains admitted work
within `--graceful-timeout`.

Identity-scoped guardrails are optional and default-off. Policies are keyed by
organization and identity. See `docs/reference/gateway-guardrails.md` for
policy lookup, the internal classifier seam, and the input and output
enforcement order.

## Authority and management

`exp config gateway` owns explicit local setup. Its provider, identity, key, grant, alias, pool, and
monthly budget commands produce versioned receipts suitable for interactive or non-interactive
callers. There are no runtime seeds. A usable installation requires an organization, active
identity, active virtual key, explicit identity-to-alias grant, active alias revision, immutable
catalog snapshot, and a resolvable provider credential reference.

Private serving authority lives in `ROOT/gateway/gateway.db`, including identities, keys, grants,
provider connections and revisions, aliases and revisions, attempts, and usage. SQLite uses WAL
mode, versioned forward
migrations, private backups before migration, newer-schema refusal, and serialized initialization.
Virtual keys are stored only as peppered fingerprints. Key material is delivered once in a JSON
receipt or to a new mode-`0600` file, and commit ambiguity preserves recoverability. Provider
configuration stores an environment variable name, never its value. Pasted provider keys live in
the user-data credential file and are resolved after a non-empty environment override. The local
pepper is mode
`0600` and is not exported.

Every data-plane request is authenticated and authorized before request decoding, routing,
continuation lookup, or provider work. Authorization freezes organization, identity, API surface,
alias revision, target, catalog digest, request digest, optional hashed operation identity, and one
monotonic deadline. Identity disable, key revocation or expiry, grant removal, and alias revision
changes fail closed.

## Catalog, aliases, and exact-model pools

The gateway database owns current provider connection state. Existing model metadata remains the
authoring input for builds, policies, evaluations, and datasets; it is not consulted as mutable
serving authority after an alias revision binds exact connection revisions. Gateway snapshots under
`ROOT/gateway/catalog-snapshots/` are immutable, secret-free artifacts for an exact catalog digest.
An advertised alias is executable only when readiness holds for the exact tuple of alias name,
alias revision, and catalog digest used by authorization.

An alias targets either:

1. a direct exact-model pool, or
2. one immutable project activation.

A singleton alias creates a one-deployment pool. `exp config gateway pool certify` can replace that
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

First-party CLI compatibility is capture-driven: the fields real Claude Code and Codex send by
default are accepted and preserved. On the Messages surface, `output_config` forwards verbatim on
Anthropic rungs (a canonical `effort` also rides `reasoning_effort`, caller keys always win over
engine-derived ones), mid-conversation `system` turns keep their position on wires that express
them (instruction-hoisting rungs narrow out), and `thinking.display` rides the verbatim thinking
config. The conditional Claude Code fields `diagnostics` and `speed` forward verbatim on
Anthropic rungs with their required `anthropic-beta` tokens and drop with disclosure elsewhere.
A caller `anthropic-beta` header forwards through an exact token allowlist (notably
`context-1m-2025-08-07`, which activates the provider's 1M context window; without it the
provider serves 200K); non-allowlisted tokens drop with a per-token
`anthropic-beta.<token>` disclosure, never a rejection and never a blind forward. On the Responses surface, `client_metadata` and `text.verbosity` forward on native rungs
and drop with disclosure elsewhere; Codex-native input items (`additional_tools` tool namespaces,
`custom_tool_call`/`custom_tool_call_output` freeform history) and non-function top-level tool
declarations (`custom` freeform-grammar tools, `namespace` tool trees, `web_search`,
`tool_search`) carry byte-for-byte at their caller positions and require a homogeneous native
Responses route; echoed message items accept `id`/`phase` with `status`
optional (non-assistant identity drops); and freeform custom tool calls stream end to end with
their native event names, including continuation retention. The item-level `namespace` on
`function_call` (plus the `name`/`namespace` pair on `function_call_output` and the
`custom_tool_call` namespace) round-trips verbatim through decode, the client stream, and
continuation retention: the provider rejects a namespaced call replayed without it, so the
field joins replay identity when present and absent items keep their exact pre-existing shape.

Provider client-errors stay sanitized: no provider error prose or body content ever reaches the
caller. The one provider-derived fact a 4xx rejection may relay is the parameter path the provider
named, extracted per dialect (OpenAI `error.param`; Anthropic's leading `path:` message token;
Gemini `google.rpc.BadRequest` field violations; Bedrock never) and only when it validates against
a strict path grammar. The path surfaces as `param` in OpenAI-shaped envelopes and folds into the
message as `(param: ...)` on the Anthropic surface; anything unextractable keeps today's
content-free message.

Before each physical dispatch, the same immediate SQLite transaction reserves the request's
conservative maximum integer micro-USD cost and inserts its attempt row. Applicable hard limits can
cover the local team, one identity, one alias pool, and each provider deployment within that pool.
An exhausted deployment allocation removes only that route from the current certified waterfall.
If no route can fit the shared team, identity, or total pool allocation, the neutral protocol
returns HTTP 429 with OpenAI `insufficient_quota` semantics before provider work. Any required
unknown price makes that route ineligible while a hard limit applies.

A deployment's price schedule may declare a long-context tier: a whole-request premium applied
once provider-reported input tokens reach its threshold, matching both published tier schedules
(Gemini reprices `prompts > 200k` entirely; Anthropic's Claude 4.6+ models serve the 1M window at
standard pricing and carry no tier). Reservation prices the tier fail-safe through the canonical
byte bound (bytes never undercount tokens), settlement selects the frozen schedule by actual
input tokens, and a tier missing a required rate keeps threshold-crossing attempts honestly
unpriced. The wait for each attempt's first provider byte scales with input size (a flat base
plus seconds per million approximate input tokens, both serving defaults with per-deployment
overrides), so a 1M-token prefill is not misread as a dead lane while small requests keep the
fail-fast bound.

Settlement replaces the reservation with observed integer micro-USD usage. A dispatched failure,
cancellation, or crash without trustworthy usage retains its conservative reservation because it
may be billable. Retries and fallbacks therefore consume one allocation entry per physical attempt,
while keyed replay creates no new reservation. A period is the immutable UTC bucket beginning at
`YYYY-MM-01T00:00:00+00:00`; rollover selects a new bucket and never clears or rewrites an earlier
month. Management and remaining-allocation reports are CLI surfaces only. There is no budgets
dashboard.

Normalized usage follows OpenAI subset semantics on every wire: `reasoning_tokens` counts a subset
of `output_tokens` and `cached_input_tokens` a subset of `input_tokens`, and settlement prices the
subset at its own rate and the remainder at the base rate. Wires that report reasoning outside
their output total are folded by the native usage mappers before the counts leave the data plane:
Gemini `thoughtsTokenCount` is additive by Google's definition and always folds into
`output_tokens`; on the OpenAI-shaped wires (Chat Completions and Responses) the provider's own
`total_tokens` decides: `input + output` is the subset shape (OpenAI, OpenRouter, Fireworks,
DeepSeek) and passes through untouched, `input + output + reasoning` is the additive shape (xAI,
natively or relayed by Azure Foundry) and folds; without a decisive total, a reasoning count above
the output total folds. Anthropic and Bedrock bill thinking inside their output total and publish
no separate count, so their reasoning subset stays unknown. The customer-visible `completion_tokens`
and `total_tokens` therefore match what is billed. Note that an additive provider's `max_tokens`
bounds only its visible answer, so a folded output total can exceed the caller's cap that the
reservation ceiling was computed from; settlement charges the exact folded total.

Each physical attempt records its own provider, model, usage, latency, terminal state, estimated
cost attribution, and frozen credential-ownership billing source. Later catalog activation and
process restart never rewrite that source. Schema-v1/v2 attempt rows migrate explicitly as
`customer_managed`; current dispatches persist either `host_managed` or `customer_managed` before
network work. The public usage report conserves physical attempt, token, cost, unknown-cost, and
terminal totals across those source buckets without partitioning logical request counts. The parent
request terminalizes once after success, final failure, cancellation, disconnect, or crash
reconciliation. Unknown prices remain unknown instead of being treated as zero or copied across
deployments.

## OpenAI-compatible protocol

`exp/runtime/openai_protocol` is the only OpenAI wire implementation. Chat Completions and
Responses have separate allowlist decoders and field-specific OpenAI error responses, but both
convert to one canonical gateway request without conflating their wire contracts. The package also
owns headers, response assembly, SSE framing, tool-call reconstruction, and official SDK
compatibility.
Chat streaming emits valid completion chunks and one `[DONE]`. Responses streaming emits the
created, in-progress, output, and exactly one terminal lifecycle. Provider tool-argument fragments
are accumulated in original order and validated only at the complete-call boundary.

`exp/runtime/anthropic_protocol` is the only Anthropic Messages wire implementation, serving
`POST /v1/messages` for Anthropic SDK callers over the same canonical gateway request. Callers
authenticate with `x-api-key` (the Anthropic SDK default) or a standard Bearer header; both carry
the same virtual key, and every failure on this surface is rendered in the Anthropic error
envelope `{"type": "error", "error": {...}}`. The decoder translates text, `tool_use`,
`tool_result`, `thinking`, and `redacted_thinking` blocks faithfully, and carries the caller's
`context_management` object verbatim (shallow-validated as an object; Anthropic rungs receive it
byte-for-byte together with its required `anthropic-beta` token, while non-Anthropic routes drop
it with `ignored_parameters` disclosure) (thinking history rides an
opaque provider-reasoning carrier with byte-exact signatures, and a caller `thinking`
configuration is forwarded verbatim on models that honor it, overriding the catalog's
adaptive default; on the adaptive-only generation, which rejects `enabled`/`disabled`
configs outright, an `enabled` config translates to adaptive with the dropped
`thinking.budget_tokens` disclosed as ignored, and `disabled` is rejected by name
because those models cannot turn thinking off), requires
`max_tokens`, validates `cache_control` (carrying it everywhere the Anthropic wire caches
natively: `tool_use` blocks, tool definitions, the top-level automatic marker, and block-level
markers on system and message text runs and on `tool_result` breakpoints all forward verbatim.
Marked runs re-emit the caller's exact block structure while the flattened string stays the
canonical content for every other wire and for digests; markerless payloads stay byte-identical.
This is what makes Claude Code sessions cacheable at all: it marks its system blocks and
conversation breakpoints on every request, and flattening them once billed whole sessions
uncached at ~10x. Responses report both cache legs back out of the folded ledger total, so
callers see `cache_creation_input_tokens` on the writing turn and `cache_read_input_tokens` on
later turns. Routes with no Anthropic rung disclose the dropped markers through
`ignored_parameters`), carries the provider-native tool annotations (`strict`,
`eager_input_streaming`, `defer_loading`, `allowed_callers`, `input_examples`; each accepted
bare by the live API, verified 2026-08-30) and `inference_geo` verbatim on Anthropic rungs with
disclosure-drops elsewhere, keeps every official SDK tool and top-level field a recorded
decision behind an SDK-surface drift gate in
`exp/runtime/anthropic_protocol/manifest.py`, and carries user `image` and PDF `document` blocks
as typed content parts (base64 or URL source, optional `title`, cache marker) that admission
checks against the route's `supports_image_input` / `supports_pdf_input` (and the `_url_input`
variants for remote sources) before dispatch, so a rung that cannot carry the attachment
rejects it loudly instead of answering from the surrounding text. Anthropic server tools are
decided per type by the same manifest: verified `web_search_*` entries forward verbatim after
the converted custom tools, their streamed output (`server_tool_use`, `web_search_tool_result`,
citation-bearing text blocks, and the `pause_turn` stop reason) reaches the caller intact on
both response paths, and a next-turn echo of those blocks (each carried verbatim as a
whole-message block) re-serves byte-for-byte; every other Anthropic-defined tool type is
rejected by name because the data plane does not yet carry its result blocks. Like the thinking
carriers, server tools replay only on the Anthropic wire, so a route with any other rung rejects
them by name instead of dropping a requested capability. The terminal `message_delta` usage
report supersedes the `message_start` input legs when present, because server-tool turns re-read
fetched results as input and the start-frame count severely undercounts the billed total.
Thinking carriers replay only on the Anthropic wire, so route admission requires every waterfall
rung to speak the `anthropic_messages` dialect; on the Responses surface over Anthropic routes,
thinking text is projected onto the reasoning-summary channel (signatures deliberately dropped)
so callers receive the reasoning they pay for, while the Chat surface has no reasoning
representation and drops it like summary deltas. Streaming emits the Anthropic
lifecycle (`message_start`, `ping`, content blocks, `message_delta` with the mapped stop reason
and usage, `message_stop`, or one terminal `error` event); the non-streaming body is the
Anthropic message object. Completed streams stop with `end_turn` (`tool_use` when tool calls are
present) and token-limited streams with `max_tokens`. The Anthropic protocol defines no
idempotency header, so this surface never joins the keyed replay stores.

Exposure-gated reasoning rungs (Tencent Hunyuan, DeepSeek — rows the catalog stamps
`reasoning_output_exposed`) return the model's plaintext `reasoning_content` on every non-tool
Chat turn, and the caller may echo that text back verbatim on later assistant turns: the
decoder carries it as an `exposed_reasoning_content` block, route narrowing forwards it only to
rungs that expose their reasoning (a route with no exposing rung rejects it by name as
`messages.reasoning_content`; a mixed waterfall prefers the exposing rung and discloses the drop
on the others), and the payload builder writes it back onto the wire unchanged. The provider's
own API accepts and does not validate that text, so it is ordinary caller-owned history, exactly
like a prior assistant `content`. A TOOL turn's reasoning still round-trips only as the sealed,
rung-pinned carrier (`x-experiential-hunyuan-reasoning-v1:`), which the same decoder recognizes
by prefix. This is what lets a Terminus-style loop (commands parsed from assistant text, output
fed back as user messages) and Harbor's interleaved-thinking replay both preserve thinking.

Route admission preserves caller capabilities in three verbatim-preference layers before any
coercion: operationally dead rungs are skipped (`dispatchable_route_profiles`), generation
controls narrow the waterfall to the rungs that preserve every exact value
(`compatible_generation_parameter_profile_indexes`), falling back to the rungs that can serve the
request only through a disclosed drop when no rung preserves it, and each remaining deployment
passes the capability preflight plus payload build. Only when zero rungs survive does the
capability-preservation policy (`exp/runtime/models/providers/capability_policy.py`) attempt one
minimal COERCE-WITH-DISCLOSURE: a reasoning effort snaps to the nearest level any rung supports
on the canonical ladder (ties prefer the lower level), ANY effort on a route with no reasoning
support at all drops (first-party clients pin effort globally, so a named rejection made whole
sessions unusable against non-reasoning models the provider itself serves fine without the
parameter; the Messages surface's verbatim `output_config.effort` is stripped with it so the
dropped value reaches the provider through no channel), and `strict: true` tools degrade to
best-effort schemas. On a reasoning route that accepts sampling only at `reasoning_effort=none`
(`sampling_requires_reasoning_none`, e.g. gpt-5.6-sol/luna), a `temperature`/`top_p` sent with
reasoning on is dropped and disclosed as `temperature->dropped(set_reasoning_effort_none)` rather
than rejected — the model accepts sampling, just not at that effort, so the request serves and the
caller is told how to keep the value (set `reasoning_effort=none`); a route that never declares the
control at all (Anthropic constrained `[1,1]` sampling) still hard-rejects it, since there is
nothing to honor at any effort. `top_k` follows the same honor-or-narrow shape: selection prefers a
rung that carries it, and a committed route with no supporting rung (an Azure `openai_deployments`
DeepSeek rung rejects it upstream) drops it with `top_k->dropped(unsupported_by_provider)` rather
than rejecting, since a rung's default sampling still returns a valid answer. `frequency_penalty`
and `presence_penalty` are admitted at the ingress and adapted the same way: honored (emitted) where
every rung supports them (the per-rung `supports_frequency_penalty`/`supports_presence_penalty`
capability truth), dropped as `frequency_penalty->dropped(unsupported_by_provider)` where a rung does
not — a soft preference whose absence still returns a valid answer. `top_logprobs` stays rejected
(not admitted): the gateway response contract does not project logprob arrays yet, so it cannot be
honored on any rung and silently dropping a probability request is never acceptable — the reject is
the honest terminal until output normalization emits logprobs. A caller
`response_format: {type: "json_object"}` is TRANSLATED, not dropped: it is admitted at the Chat
ingress and rewritten to a permissive non-strict `json_schema` (`{"type":"object"}`, "any JSON
object") — the serving lanes emit only `json_schema`, so this preserves the caller's JSON intent on
every rung (dropping it would hand prose to a caller who asked for JSON) — and disclosed as
`response_format->translated(json_object)`; a non-strict schema is left open (never force-closed to
`additionalProperties:false`), so its "any object" meaning is not inverted on a schema-closing
(Anthropic) rung. A caller `service_tier` on the OpenAI-family surfaces forwards verbatim
only on rungs dispatching tenant-owned (BYOK) credentials, where the caller pays the provider
directly; host-funded rungs never emit it (the tier changes provider pricing while the gateway
bills catalog rates) and a route with no eligible rung drops it with disclosure. Anthropic's own
`service_tier` stays a recorded Messages-surface rejection. On `maximize_cache` pools, a cache-marked request dispatches
marker-honoring (Anthropic Messages) rungs before marker-dropping wires, stably within each
group, so a shim rung can no longer silently bill every turn's full context uncached while the
native rung stands ready; routes narrowing to only marker-dropping wires keep disclosing the
dropped markers. Every coercion is disclosed in `path->effective` form
through `ignored_parameters`, logged, and counted in the `admission_parameter_coercions`
metric; every serving surface carries that list to the caller as a body-level
`x-experiential-ignored-parameters` key (Chat chunk and completion, Responses envelope, and the
Anthropic message on both `message_start` and the aggregated body), so a drop is never silent; nothing coercible keeps the first rung's own field-scoped rejection.
The per-deployment `capability_parity` export joins catalog declarations with the engine's
provider-family ground truth so a catalog can pre-warn on gaps and route around them before a
caller hits that 400.

Commit-independent headers are available before streaming begins. Route-dependent headers are
emitted only after an execution snapshot exists. Stable public IDs do not expose raw key,
idempotency, request, or provider values.

`GET /v1/models` lists only the aliases granted to the presented key. The envelope contains
only the OpenAI `object` and `data` fields, and every entry contains only `id`, `object`,
`created`, and `owned_by`. Capability, pricing, revision, and catalog-digest metadata never
ride this compatibility endpoint. Platform's separate `/api/models` catalog owns rich route
metadata, including configured micro-USD-per-million-token prices.
`GET /v1/models/{model_id}` describes one granted alias with
the same exact OpenAI Model object and
returns the identical `model_not_found` 404 for every other model ID, so the route never confirms
whether an ungranted alias exists. Quota-exhausted and throttled 429 responses and the draining
503 advertise a `Retry-After` wait, and monthly quota exhaustion reports its exact UTC
calendar-month reset boundary in the error message.

OpenAI `3.0.0` `OpenAI` and `AsyncOpenAI` clients are release-certified for Chat Completions and
Responses in synchronous and asynchronous, streaming and non-streaming forms. Responses
continuation and duplicate replay retain content only in bounded, process-local, tenant and
alias-revision-scoped stores. A `store: false` request skips continuation retention entirely
(continuing from its ID answers `continuation_unavailable`), and
`include: ["reasoning.encrypted_content"]` forwards the encrypted reasoning request to native
OpenAI Responses routes, whose opaque payloads replay verbatim from the caller's input; the
replayed reasoning item's `id` is never forwarded upstream because the provider binds the
encrypted payload to its original item id and callers echo this gateway's own minted public ids. Replay is opt-in through the standard `Idempotency-Key` header only;
`X-Client-Request-Id` is caller correlation identity (Codex sends its session id there on
every request of a session), echoed on responses and used for route affinity, never as an
operation key. Restart
or eviction returns an explicit unavailable error and never reconstructs content from SQLite.

## Content-free observability and lifecycle

SQLite stores hashes, frozen authority, route identity, state transitions, token counts, latency,
and estimated cost. It never stores prompts, responses, raw tool arguments, raw virtual keys, or
provider secrets. `GET /usage` and `GET /usage.json` are two renderings of the same schema-v2 report
and expose only aggregate, per-identity, and physical-attempt `by_billing_source` accounting.
An anonymous request reads the organization-wide report; a request carrying a virtual key as
`Authorization: Bearer <key>` reads the report scoped to that key's identity, and an invalid
key is rejected with the standard 401 error.
Source buckets conserve attempt, token, known-cost, unknown-cost, and terminal-state totals but do
not partition logical request counts. Estimated cost is attribution, not a provider invoice.

The process owns readiness from preflight through bounded drain. New work is rejected after drain
starts. Admitted tasks, upstream streams, disconnect cleanup, replay ownership, continuation state,
and final ledger settlement are process-owned and bounded. A stuck cancellation cannot prevent the
terminal flusher from attempting content-free settlement.

## Certification boundary

Deterministic release evidence uses a built and freshly installed wheel, real SQLite, a real
subprocess-bound loopback gateway, a real loopback upstream, and the official SDK clients. One
scanner checks database, WAL, backups when present, catalog snapshots, stdout, stderr, logs, usage
responses, and error bodies for raw content and secret canaries.

`exp/runtime/gateway/provider_certification.py` is the dated provider capability matrix. Each cell
names the official client SDK, public gateway surfaces, provider wire surface, fixture result, and
credential-gated live status. OpenAI and Anthropic have native fixtures; generic OpenAI-compatible,
Azure, and OpenRouter share compatible-stream coverage; Gemini and Bedrock have native deterministic
fixtures. Live provider cells remain explicitly `not_run_requires_credentials` until a separately
authorized run supplies dated evidence. Deterministic fixtures do not imply hosted-provider
availability, billing, or account-specific behavior.

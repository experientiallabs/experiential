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
- `POST /v1/messages` (the Anthropic Messages API) and `POST /v1/messages/count_tokens`, which
  answers Anthropic's `{"input_tokens": N}` from the gateway's own reservation tokenizer (the
  credit reservation's estimator without its headroom): authenticated and granted exactly like
  `/v1/messages`, no ledger row, no charge. The gateway has no tokenizer authority for any rung
  (the Anthropic provider client does not forward `count_tokens`), so every answer is an
  estimate and the body says so through the shared `x-experiential-ignored-parameters`
  disclosure (`input_tokens->estimated(gateway_tokenizer)`).
- `POST /v1/embeddings` (the OpenAI Embeddings API: message-less and never streamed; served
  only by aliases whose catalog capabilities declare `supports_embeddings` on an OpenAI-wire
  connection, billed on the provider's reported `prompt_tokens` with no output leg, and
  returned with the provider's exact vectors in `float` or `base64` form. Input accepts a
  nonempty string, a batch of nonempty strings, a nonempty integer token sequence, or a
  batch of nonempty token sequences. A flat token sequence produces one vector; token IDs
  are forwarded unchanged and must match the selected model's tokenizer. Boolean, fractional,
  negative, mixed-shape, and empty token inputs are rejected. Omit `stream` or send the literal
  `false`; `true` and unknown parameters are rejected, and `stream` is never forwarded.
  Reservations count token IDs directly and retain normal headroom; settlement still uses
  provider-reported input usage. There is no response cache, and an inbound `Idempotency-Key`
  is ignored because the surface has no replay protocol)
- `POST /v1/images/generations` (the OpenAI Images API, generations only: prompt in, images
  out, never streamed; served only by aliases whose catalog capabilities declare
  `supports_image_generation` on an OpenAI-wire connection, billed on the provider's reported
  prompt and image tokens, so a model that answers without token usage is refused as
  unbillable rather than served for free)
- `POST /v1/systemone` (TypeSafe native decisions: typed `noul`, `choice`, and `score`
  questions, buffered answers, provider-reported usage, and explicit decision capability and
  pricing admission; no chat, streaming, continuation, or idempotency replay)
- `GET /health/live` and `GET /health/ready`
- `GET /usage` and `GET /usage.json`

`exp --project PROJECT` is compatibility sugar that activates one project-backed alias and launches
this same gateway application. It does not create a router HTTP server. Gateway startup and readiness
perform no provider request. Only an authorized model request may cross the provider boundary.

Chat Completions and Responses tool-call IDs are opaque strings of 1 to 65,536 characters.
Replay each complete ID, including any signature suffix, in both the assistant call and tool result.
OpenAI-compatible Chat routes preserve IDs verbatim; other API dialects may restrict their wire shape.
Output guardrail byte limits count the full serialized completion, including tool IDs, names, arguments, and JSON framing.

Streamed function-call arguments must assemble to one JSON object. On OpenAI-compatible
Chat streams the gateway stops relaying argument deltas at the byte that closes that object:
whatever the provider streams after it is withheld and judged at completion. A tail that is
only whitespace, an exact repetition of the whole object, or (after a zero-argument `{}`)
empty literals such as `""` is dropped and the call completes — Azure Foundry's DeepSeek shim
streams `{}` then `""` for every zero-argument call — so the deltas a client receives always
concatenate to the completed call's bytes. Any other tail, and any syntax error inside the
object, keeps the strict contract: the attempt fails as `malformed_response` (a provider
fault, eligible to fail over to a later deployment), the ledger names the parse position and
byte count, and the operator log names the tool (bounded to an identifier token, since a
relay can put arbitrary model output in the name field); argument bytes are never logged or
repaired by guessing.

Arguments that END mid-value (valid JSON so far that simply stops) are the provider's cut,
whatever it declared about the ending: a Chat relay's `stop`/`tool_calls` finish, OpenAI's
own Responses `function_call` item marked `completed` (gpt-5.6-luna, exp#896), a Bedrock or
Anthropic block stop followed by `tool_use`/`end_turn`, or a stream that closes without its
terminal frame. A model never ends a well-formed call mid-string, so every dialect drops the
cut call (operator log `tool_arguments_cut_mid_fragment` with what the provider declared) and
the turn settles `incomplete` (`finish_reason: length`), exactly as a provider-declared
`max_tokens` truncation does; the caller's remedy is a larger budget, never a retry of a
"malformed" provider. Only a syntax error INSIDE the arguments is corruption and stays
`malformed_response`.

Two relay shapes decode leniently instead of failing: a tool call streamed with a null or
empty `id` gets a gateway-minted id (`call_gw<index>_<clock>`; a real id restated later is
ignored, since the caller already holds the minted one), and an entry with an empty name
and no arguments is a placeholder, dropped without starting a call (a nameless entry that
does carry arguments still fails: a name cannot be invented). A Chat frame with `choices` absent
or null and no content-bearing key (`delta`, `message`, `finish_reason`, `error`, …) is a trailing
metadata chunk (usage, Novita's `sla_metrics`) and is read as such; one carrying content stays
malformed, naming the frame's key names. A Responses reasoning-summary `done` text that differs
from its relayed deltas is logged, not failed (the summary is display-only prose).

A stream that closes cleanly WITHOUT its terminal frame is judged by what it served: before
any output it is `provider stream ended without a terminal event` (failover-eligible, nothing
to preserve); after output, Gemini completes (its documented shape), an OpenAI-compatible
relay that already declared its finish settles by that finish, and every other wire settles
`incomplete` with open items closed and any mid-fragment call dropped (operator log
`stream_ended_without_terminal_after_output`).

## The data plane

The gateway has one data plane: the native Rust HTTP server in the PyO3 extension
`exp_gateway_native`. Every launch uses it; a missing extension fails with its build command,
never a Python fallback. Sockets, upstream dispatch, normalization, and SSE encoding run off the
GIL. JSON-string callbacks authenticate and admit requests, reserve each physical dispatch through
`start_attempt`, and settle outcomes. `enforce_output` runs only when admission sets
`output_guardrail`; unguarded and non-chat decision requests do not call it.
Python owns surface-specific decoding, authority, exact deployment identity, payload construction,
and durable SQLite transactions over hot-reloadable authority generations. Chat uses `decode_chat`
and the `streaming_requests` builders; decisions use `decode_decision_request` and their typed body.
Wire facts come from each resolved client's `gateway_wire_profile()`. The dialects are
`openai_responses`, `anthropic_messages`, `openai_compatible` (including Azure and OpenRouter),
`gemini_generate_content`, `bedrock_converse_stream`, and decision-only `typesafe_systemone`.
Bedrock uses AWS binary event streams, not SSE. Admission freezes the Converse body; after its
bounded dispatch permit, the data plane obtains SigV4 headers through Python's `sign_dispatch`
callback immediately before POSTing those exact bytes. Signing after queue wait avoids stale
signatures. The bounded immediate open retry reuses that signature; later retries sign afresh.

### Native decisions

`POST /v1/systemone` accepts only `model`, `state`, and named `questions`; see
[TypeSafe SystemOne decisions](providers.md#typesafe-systemone-decisions) for all three typed
question and answer shapes. Authentication precedes decoding; authorization and durable acceptance
precede direct-route resolution. Admission requires `deployment.gateway.capabilities.supports_decisions`
and the `typesafe_systemone` endpoint, plus a known nonnegative input rate and output rate exactly
zero. Missing capability, a project target, or unsupported pricing fails closed before dispatch.
The decoder bounds requests to 32 questions, 64 choice or 10 score criteria, and 262,144 bytes.
Reservations count repeated state and per-question protocol allowances, not the chat tokenizer.
They are bounded estimates, never provider-enforced token ceilings. Only reported usage settles.
Only HTTP 400/401/403/404/422 release a known-rejection hold; 401 may use a certified fallback.
HTTP 402/429/529, ambiguous transport, and malformed answers are terminal unknown outcomes, holds kept.
At most eight deployments run once each: no redials, chat, streaming, replay, or chat guardrails.

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

Conversational guardrails are optional and default-off, keyed by organization and identity.
See `docs/reference/gateway-guardrails.md` for policy lookup, classifiers, and enforcement order.

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

Authentication precedes body decoding; authorization precedes routing and provider work. It freezes
organization, identity, API surface, alias revision, target, catalog and request digests, optional
hashed operation identity, and a monotonic deadline. Disabled identities, revoked or expired keys,
removed grants, and alias revision changes fail closed.

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

Tool descriptions have no per-field character limit and reach providers unchanged. The 64 MiB
request-body limit and provider context limits still apply. Gateway accounting and diagnostic logs
remain content-free. A downstream host that opts into logging tool descriptions must cap each
logging-only preview at the first 262,144 Unicode characters and record the original character
count and whether truncation occurred. This policy applies only to diagnostic copies. Provider
payloads contain the complete description, and request identity hashes include its full content.
Durable gateway request records store that digest without retaining the description itself.

The content-free ledger accepts the logical request before learned project selection. Selection or
direct resolution then produces an execution snapshot containing the exact model, pool, and ordered
deployment IDs. Each physical provider dispatch gets its own durable attempt row immediately before
network work. Attempt ordinal counts all physical dispatches; route depth identifies the selected
deployment position.

Conversational provider execution is internally streaming. Bounded same-deployment retries and ordered
deployment fallback are allowed only for typed precommit failures. The first outward text, refusal,
or tool-call semantic event commits the deployment, after which the gateway never switches
providers. Typed refusal fallback is disabled unless the active alias revision explicitly enables
it. Opted-in refusal deltas are withheld only in a bounded in-memory buffer: a refusal-only terminal
result can advance to the next certified deployment, while mixed semantic output or buffer overflow
commits and flushes the original route. A `stop` that bills output or reasoning tokens yet carried no
semantic event (a stripped reasoning-only turn, OpenAI's 4-token empty message) is a typed
`empty_completion` failure ($0, outside the health circuit): it redials once, takes the ladder, and an exhausted
ladder answers a typed 200 under `x-gateway-warning: empty_completion` (image-output rungs skip redial + ladder); a zero-token stop
that REPORTS zero tokens and a budget truncation before the first delta stay honest output-less
answers. A `stop` with no semantic event and NO usage report at all is read by the caller's own
output cap (the admission carries `maximum_output_tokens`): on a capped request it is a budget the
provider's private reasoning exhausted before the first visible token, mislabelled as a plain stop
(Meta muse-spark under a small `max_tokens`), and answers `length` / `max_tokens` / `incomplete`
with the ledger settled `incomplete`; on an uncapped request it is the provider delivering nothing
and takes the same ladder as the billed empty stop. The Messages surface
applies the same rule after commitment, when every committed event was one it cannot render.
Provider-internal retry layers are disabled so every
possible billable dispatch is visible to the gateway ledger.

**Per-rung conditional failover (`failover_only_on`).** A deployment may restrict itself to
failover duty for a named set of failures (a customer's trusted-access OpenAI key taking only the
house rung's `refusal:cyber_policy`); see [gateway-failover-rules.md](gateway-failover-rules.md).

A provider throttle (HTTP 429, an overload answer, or a rate-limit error declared inside the
stream) is classed `throttled` and is failover-eligible but never redialed on its own: the 429
sets the rung's throttle window before the next candidate is chosen. What happens next is the
pool's per-model policy. `failover_mode: maximize_availability` (the default) advances to the next
certified rung; `maximize_cache` returns the 429 to the caller so it can retry the warm rung after
the provider's backoff instead of restarting cold; `maximize_cache_affinity` advances to the
deterministic rendezvous alternate. An authored `throttle_cache_threshold` (0..1) replaces the
mode's fixed rule under every mode: the throttle surfaces exactly when the requesting
organization's observed cached-token fraction on the throttled rung (the worker's EWMA of its
settled `cached_input_tokens / input_tokens` there) is at or above the threshold, else fails over
cold, disclosed on the cold attempt as `throttle_failover_cold` with the throttled rung as its
`preferred_deployment_id`.

A pool may additionally author a `throttle_redial` schedule (`max_attempts`, `base_delay_ms`,
`max_delay_ms`). With it, a throttle is never returned while any way to serve the request
remains. Before commitment, a throttle on a rung worth waiting for is re-dialed on the SAME
deployment after a bounded wait: `base_delay_ms * 2^n` (equal-jittered into the upper half of the
band, capped at `max_delay_ms`), floored at the provider's `Retry-After` when it states one within
the cap; a `Retry-After` above `max_delay_ms` means the rung is out longer than the pool will
wait, so the ladder advances instead. A wait never exceeds the rung's first-byte allowance and
must leave the redial its own first-byte allowance under the request deadline; otherwise the
ladder advances at once. Up to `max_attempts` redials per rung are made, then the throttle fails
over down the ladder exactly like any failover-eligible failure, and only when every rung is
exhausted does the typed 429 reach the caller, carrying the largest `Retry-After` any rung stated.
How long each rung is worth waiting on is decided at admission per rung and carried on the wire
entry as `throttle_redial_budget`, the post-backoff redials this request may spend there: the
full `max_attempts` on every rung when no `throttle_cache_threshold` is authored; otherwise three
rules apply per rung, in order. (1) No cold alternative: the last rung of the admitted route (the
route already narrowed to rungs that are live and can serve the request, so a single-rung route is
the same case) gets the full `max_attempts` regardless of cache evidence, because a throttle only
advances cold to later rungs and a zero budget there would surface the 429 while a bounded wait
could still serve. (2) Warm sticky session: a rung the request's affinity fingerprint holds a live
worker-local sticky binding to gets the full `max_attempts`, the binding being direct evidence that
the conversation's provider cache lives there. (3) Otherwise the budget scales with the cache at
stake: the full budget where the organization's cached fraction meets the threshold, a
proportional share (`floor(max_attempts * fraction / threshold)`) below it, and zero with no cache
evidence, so the gate means "how long to wait here" rather than "surface": high stake spends the
whole backoff budget, low stake fails over sooner, no stake fails over at once. Rules 1 and 2
exist because the fraction is a worker-local EWMA that reads zero on any worker without a recent
settled sample from the organization, which at a few requests per hour spread across workers is
most of them, even for a conversation that is over ninety percent cached at the provider; missing
evidence must never strand a request on a pool whose operator asked for backoff. Every redial is its own durably reserved
attempt row (the attempt ordinal increments), claimed through the rung's own throttle window
(this request is the one deliberately probing the rung back; other requests still avoid it), and
disclosed as `dispatch_reason: throttle_backoff`; the cold advance after the budget is `throttle_failover_cold`. The redial is admitted on the warm rung even when that rung's own per-worker RATE WINDOW (`requests_per_minute` / `tokens_per_minute`, the `rate_limit` shed) would shed it: the per-minute windows are pacing, and a redial that already waited the backoff has paid its pacing on the provider's own 429 clock, so the shed is force-admitted (attempt row `throttle_backoff`, shed still counted in `rung_admission_sheds`, forced redial in `throttle_backoff_forced_admissions`) rather than converted into a cold failover of a prompt a fallback may never finish within its first-byte allowance. The rung's `concurrency_bound` (`queue_bound`, `fresh_session_spill`) and its fair share are NOT bypassed: the bound is the per-worker hard ceiling that protects the provider connection and other tenants, so a redial shed by it spills sideways like any dispatch. Sheds on other rungs and of a first dispatch spill sideways as without a schedule; a forced redial the deployment budget then rejects carries no forced state to the next rung. The worker's
`throttle_backoff_redials` counter beside `throttle_surfaced_cache_preserving` and
`throttle_failover_cold` traces the three outcomes. Pools
that author no schedule keep byte-identical behavior; the hosted platform's recommended
authoring for house GPT lanes, whose traffic is cache-heavy, is a schedule of three redials from a
500 ms base capped at 8 s beside its existing 0.5 threshold.

A reasoning continuation (a Chat request replaying a gateway-sealed `reasoning_content` carrier
on an assistant tool turn after the latest user message) resolves as `route_reason:
reasoning_continuation`: the carrier is authenticated against the exact deployment and credential
that sealed it, and that issuing rung dispatches first with the unsealed reasoning replayed, so
the model's thinking continues across the tool call. The rung is NOT the whole ladder. The pool's
other certified rungs follow in pool order as failover fallbacks, and every one of them is frozen
at admission from the request with the post-user-boundary sealed reasoning removed: only the
issuing rung's credential can unseal a carrier (each provider's carrier is AEAD domain-separated
to its own credential, a non-carrier rung yields no authority, and the payload builders reject a
block sealed for another route by name), so the fallback keeps the messages, visible text, tool
calls and tool results and drops just that turn's thinking. A failover-eligible operational
failure on the issuing rung (a throttle after the pool's `throttle_redial` budget is spent there,
provider quota, unavailability, transport) therefore advances to the next rung exactly like any
last-rung failure, and that attempt is recorded with `route_reason:
reasoning_continuation_failover`; a caller error (`invalid_request`, a refusal without the
opt-in) still surfaces without touching a fallback. The stated loss on a failover is the model's
thinking continuity across that tool call and the issuing provider's prompt cache for the turn,
never correctness of the visible conversation. Because its fallbacks run without the reasoning,
the issuing rung gets the full `throttle_redial` budget (rule 2 above, beside the sticky rung),
affinity or cache-marker reordering never demotes it from first position while it is
dispatchable, and a rung dispatch-policy shed of the issuing rung (its authored per-worker
`requests_per_minute`, `tokens_per_minute` or `concurrency_bound`, which trip under ordinary
load) force-admits it as `saturated_overflow` exactly as a one-rung ladder did instead of
spilling sideways to a stripped fallback: only a real failover-eligible failure on the pinned
rung moves the ladder past it. A continuation
whose sealed carriers all precede the latest user message carries no active reasoning and routes
as a plain request; a single-rung pool has no fallback and surfaces the failure as before.

First-party CLI compatibility is capture-driven: the fields real Claude Code and Codex send by
default are accepted and preserved. On the Messages surface, `output_config` forwards verbatim on
Anthropic rungs (a canonical `effort` also rides `reasoning_effort`, caller keys always win over
engine-derived ones); OpenRouter's `reasoning` object supplies an alternate effort or enable
channel. On Messages, a numeric `max_tokens` reasoning budget becomes an exact budgeted
`thinking` configuration on budget-capable wires; it cannot become an advisory effort or
replace another explicit numeric budget. Unsupported hard constraints are refused before
dispatch. `exclude` retains its disclosed omission policy. The Chat surface accepts effort
and enable controls beside `thinking`, `chat_template_kwargs`, and DashScope's top-level
`enable_thinking`; numeric budgets travel unchanged on qualified Anthropic, Gemini 2.5 and Qwen Cloud
routes. Unsupported budgets are refused; see [generation controls](gateway-generation.md). Chat also
replays OpenRouter's `reasoning` / `reasoning_details` (the `reasoning.text` blocks) as the
same caller-owned plaintext history a `reasoning_content` echo is; mid-conversation `system` turns keep their position on wires that express
them (instruction-hoisting rungs narrow out), and `thinking.display` rides the verbatim thinking
config. The conditional Claude Code fields `diagnostics` and `speed` forward verbatim on
Anthropic rungs with their required `anthropic-beta` tokens and drop with disclosure elsewhere.
A caller `anthropic-beta` header forwards through an exact token allowlist (notably
`context-1m-2025-08-07`, which activates the provider's 1M context window; without it the
provider serves 200K); non-allowlisted tokens drop with a per-token
`anthropic-beta.<token>` disclosure, never a rejection and never a blind forward. On the Responses surface, `client_metadata` and `text.verbosity` forward on native rungs
and drop with disclosure elsewhere. Chat `verbosity` accepts `low`, `medium`, or `high` as the
same hint: forwarded as `text.verbosity` on native Responses routes and omitted with a
`verbosity` disclosure on other routes. Invalid values remain named parameter errors.
Codex-native input items (`additional_tools` tool namespaces,
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
The SDK 3.0 programmatic tool-calling `caller` object on `function_call`,
`function_call_output`, and `custom_tool_call` gets the same verbatim round trip (validated only
as an object; its internal shape is the provider's). `function_call_output.output` accepts the
SDK list form: text and image parts map onto the canonical tool message and re-emit typed, an
all-text list keeps the plain-string wire shape, and any other part kind is a named 400. A
reasoning input item without `encrypted_content` (a `store: true` replay by item id) carries
verbatim to homogeneous native Responses routes and the provider judges resolvability. Off the
native Responses wire, tool-call and tool-result attribution (`namespace`/`caller`/output
`name`) drops with per-field disclosure — the call itself always survives — and a
Messages-surface effort the route cannot serve rejects as `output_config.effort` with
"effort parameter … not supported" phrasing, the exact predicate Claude Code's built-in
drop-and-retry recovery latches on.

Provider client-errors stay sanitized: no provider error prose or body content ever reaches the
caller. The one provider-derived fact a 4xx rejection may relay is the parameter path the provider
named, extracted per dialect (OpenAI `error.param`; Anthropic's leading `path:` message token;
Gemini `google.rpc.BadRequest` field violations; Bedrock never) and only when it validates against
a strict path grammar. The path surfaces as `param` in OpenAI-shaped envelopes and folds into the
message as `(param: ...)` on the Anthropic surface; anything unextractable keeps today's
content-free message.

Before each physical dispatch, the same immediate SQLite transaction reserves the request's
conservative maximum integer nano-USD cost and inserts its attempt row. Applicable hard limits can
cover the local team, one identity, one alias pool, and each provider deployment within that pool.
An exhausted deployment allocation removes only that route from the current certified waterfall.
If no route can fit the shared team, identity, or total pool allocation, the neutral protocol
returns HTTP 429 with OpenAI `insufficient_quota` semantics before provider work. Any required
unknown price makes that route ineligible while a hard limit applies. For conversational requests,
the input reservation is a tokenizer estimate, not a byte bound: prompt text, tool schemas,
structured-output schema, and replayed provider carriers are counted once with the o200k BPE,
inline media reserve documented planning constants instead of their base64 length, and the
total carries fifteen percent headroom plus per-message and per-tool framing. The same number
feeds the paid worst-case ceiling and the host's free-tier and token-rate windows; settlement
replaces it with the provider's exact usage.

A rung may author a `GatewayRungDispatchPolicy` on its gateway metadata (all fields inert by
default). Its `concurrency_bound` is a per-worker in-flight cap enforced by pure in-process
counters at the same pre-dispatch point: a rung at its bound is bypassed sideways to the next
claimable rung (spill in seconds) instead of queueing at the deployment until the request
deadline. `requests_per_minute` and `tokens_per_minute` (each usable without the bound) cap the
rung's sliding 60-second dispatch window the same way, shedding a reservation the window cannot
absorb sideways as `rate_limit` BEFORE the provider answers 429; token accounting counts each
dispatch's conservative worst-case reserved input plus output tokens at reservation, with a
burst allowance admitting a single over-cap reservation into an EMPTY window (a prompt whose
worst case exceeds the whole per-worker cap must stay admissible, then blocks the window until
it slides out). The working
request ceiling is additionally calibrated passively per worker: a provider throttle settlement
clamps a learned ceiling to ninety percent of the rate observed in the window at that moment,
every unthrottled recovery minute creeps it back up by five percent (at least one request, capped
at the authored rate when one exists; each creep step is the probe that rediscovers headroom, so
no synthetic traffic is ever sent), and a ceiling unthrottled for six hours is forgotten. The
learned ceiling is a float and may sit below one request per minute: per-worker ceilings
multiply across the fleet, and some provider accounts allow less than one request per worker
per minute. With
`fair_share: true` (which requires the bound), a contended rung additionally
limits each organization to its weighted max-min share of the bound; weights arrive per request
on `AuthorizationSnapshot.fair_share_weight` (default 1) from the hosted store, capacity below
the bound is always borrowable (a lone organization uses the whole rung), freed slots are
reserved for recently active under-share organizations, and running dispatches are never
preempted. With `cache_priority_alpha` authored on a fair-share rung, each organization's
effective weight becomes `weight * (1 + alpha * congestion * cached_fraction)`, where congestion
is the rung's in-flight total over its bound and the cached fraction is the worker's time-decayed
EWMA (half-life roughly ten minutes) of the organization's settled cached-token share on that
rung, so at the contended margin traffic that reuses warm provider cache is admitted ahead of
equal-weight cold traffic. A ladder whose every remaining rung was bypassed only by these
policies force-admits past the bound rather than manufacturing a failure unbounded admission
would not have had.
Every policy-routed dispatch is disclosed on its attempt row: `dispatch_reason` (`affinity`,
`affinity_sticky`, `fair_share_shed`, `queue_bound`, `rate_limit`, `fresh_session_spill`,
`rung_dead`, `saturated_overflow`, `throttle_failover_cold`, `throttle_backoff`), the bypassed
`preferred_deployment_id` with its frozen base token rates, and at settle a
`counterfactual_cost_nano_usd` pricing the same observed usage at those preferred rates, so
cost optimality is measurable from the ledger alone. Settlement also persists the provider's own
rate-limit response headers per attempt when the data plane harvests them (`retry-after` plus the
OpenAI `x-ratelimit-*` and Anthropic `anthropic-ratelimit-*` families, normalized to integers),
and a throttled settlement carrying a parseable `Retry-After` (seconds or HTTP-date) sizes that
deployment's throttle window from it, clamped to [5s, 6h], instead of the fixed default, so a
daily-quota reset actually suppresses the rung for the wait the provider asked for. Pools and
rungs that author none of this keep byte-identical behavior and null disclosure columns.

Under `maximize_cache_affinity`, two further per-rung fields keep provider prompt caches warm
across spills. `sticky_spill_seconds` gives each dispatch a worker-local
fingerprint-to-deployment binding with that lifetime (refreshed per hit, but capped at four
lifetimes of total age from creation so continuous hits cannot pin a long-running session to a
pricier spill rung forever): the binding is honored
ahead of rendezvous order on later requests, so a spilled conversation keeps serving off the rung
holding its warm cache instead of bouncing back the moment the preferred rung stops shedding, and
a binding whose rung is throttled or circuit-open is cleared rather than followed. The binding is
deliberately worker-local (the serving edge's keep-alives pin a client to one worker; the
cross-worker miss costs one cold dispatch). The binding keys on the affinity fingerprint (the
session identity rendezvous already uses), never on a derived provider cache key: on
OpenAI-compatible shim lanes (including Experiential Cloud's vLLM boxes) no `prompt_cache_key` is
forwarded and the box's prefix cache is content-addressed, so gateway-side session-to-rung
consistency is the entire cache-preservation mechanism there. `fresh_session_spill_fraction`
reserves the top slice
of a bounded rung for warm sessions: a request whose fingerprint holds no live binding on the
rung sheds sideways once in-flight dispatches reach `bound * fraction` (`fresh_session_spill`),
while warm sessions ride to the hard bound (it requires `sticky_spill_seconds`, because warm
standing IS a live binding). A hosted composition may also exclude individual attempts from the
cache-priority EWMA through the accounting's `cache_sample_gate` (promotion-funded replay must
not buy fair-share weight with prefixes the promotion already made costless).

A deployment's price schedule may declare a long-context tier: a whole-request premium applied
once provider-reported input tokens reach its threshold, matching both published tier schedules
(Gemini reprices `prompts > 200k` entirely; Anthropic's Claude 4.6+ models serve the 1M window at
standard pricing and carry no tier). Reservation treats the tier as reachable from a documented
margin below its threshold (the input reservation is a tokenizer estimate with headroom, not a
bound), settlement selects the frozen schedule by actual input tokens, and a tier missing a
required rate keeps threshold-crossing attempts honestly unpriced. The wait for each attempt's first provider byte scales with input size (a flat base
plus seconds per million approximate input tokens, both serving defaults with per-deployment
overrides), so a 1M-token prefill is not misread as a dead lane while small requests keep the
fail-fast bound.

Settlement replaces the reservation with observed integer nano-USD usage. A dispatched failure,
cancellation, or crash without trustworthy usage retains its conservative reservation because it
may be billable. Retries and fallbacks therefore consume one allocation entry per physical attempt,
while keyed replay creates no new reservation. A period is the immutable UTC bucket beginning at
`YYYY-MM-01T00:00:00+00:00`; rollover selects a new bucket and never clears or rewrites an earlier
month. Management and remaining-allocation reports are CLI surfaces only. There is no budgets
dashboard.

Normalized usage follows OpenAI subset semantics on every wire: `reasoning_tokens` counts a subset
of `output_tokens`; cache reads and writes are disjoint subsets of `input_tokens`.
`cache_creation_input_tokens` crosses the hosted settlement callback; Chat exposes it as
`prompt_tokens_details.cache_write_tokens`. Hosted settlement prices each observed subset at
its own rate; the local SQLite cost estimate still lacks a separate cache-write rate. Wires that report reasoning outside
their output total are folded by the native usage mappers before the counts leave the data plane:
Gemini `thoughtsTokenCount` is additive by Google's definition and always folds into
`output_tokens` in the native and Python Gemini paths; on Chat and Responses, the provider's own
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
Chat image references accept Copilot's optional `image_url.media_type` MIME hint
(`image/png`, `image/jpeg`, `image/gif`, or `image/webp`). The hint is validated and
discarded before provider dispatch and canonical replay identity; the `url` and
`detail` remain authoritative. A data URL keeps its embedded MIME type, and a
remote URL is forwarded for the provider to fetch. Unknown image fields and
malformed URLs or base64 remain rejected.
A Chat `role: "tool"` message accepts `image_url` parts beside its text (GitHub Copilot Chat,
Codex Desktop and node agents report a screenshot inside the tool message that took it; the
old `valid only for user messages` 400 refused ~1,000 such requests a week and wedged every
later turn of those sessions, because the block is baked into the caller's history). The
result decodes as the canonical tool message with text and image parts, the same shape the
Anthropic `tool_result` image block and the Responses `function_call_output` part list
produce; video, audio and file parts inside a tool message stay a named 400, and every other
non-user role stays text-only. Each wire then carries the image: natively inside the tool
result on Anthropic, native Responses and Bedrock (`ToolResultContentBlock.image`), and on Chat
Completions and Gemini, whose tool results are text-only, folded into ONE user message that
follows the last tool message of the contiguous run (a user message between two results of a
parallel batch breaks the provider's tool-call linkage): each tool message keeps its text with
a numbered `[image N: attached in the next user message]` marker where the image stood, the
user message opens with a fixed header and introduces each image by number and
`tool_call_id`, and the route discloses
`messages.content.tool_result.image->following_user_message`. Only a rung with no image input
at all still degrades the tool image to placeholder text with the
`messages.content.tool_result.image->placeholder` disclosure (`capability_policy`); a top-level
user image keeps the fail-closed contract because the caller can re-send it.
Both OpenAI surfaces accept the Vercel AI SDK's camelCase `promptCacheKey` (sent verbatim by
opencode and other `ai-sdk` coding clients) as an alias of `prompt_cache_key`: it is renamed
before manifest validation and decodes exactly as the documented field. When both spellings
arrive, `prompt_cache_key` wins and the dropped alias is disclosed through `ignored_parameters`.
It is the only camelCase spelling admitted; every other unknown top-level field stays a named 400.
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
adaptive default. Budgeted-thinking support and support for disabling thinking are separate
model capabilities: a valid explicit off setting is preserved, and an unsupported off setting
is refused before dispatch. A numeric thinking budget is never silently replaced with an
advisory effort. A bare `{type: enabled}` without a numeric budget can use a disclosed derived
budget that fits the effective per-rung output cap, or an admissible effort on an effort wire;
an impossible requested configuration is refused instead of disabling thinking. An explicit
budget at or above `max_tokens` is refused on `thinking.budget_tokens`. See
[generation limits and interrupted streams](gateway-generation.md) for output-cap provenance,
partial tool calls, connection-loss handling, and the accounting boundary), requires
`max_tokens` (a ceiling under 1024 with no reasoning signal of the caller's own, on a route whose
every rung reasons by default and offers a `none` tier, dispatches at `reasoning_effort: none`
disclosed as `reasoning_effort->none(max_tokens_headroom)`, so the reply is text rather than
thinking cut off at the ceiling), and validates `cache_control`. Both Messages and Chat
Completions retain message/text breakpoints on canonical carriers. Anthropic (including Claude
on Azure Foundry) forwards them, Bedrock translates them into `cachePoint` blocks, and OpenRouter
forwards them on its Chat wire. Generic Chat adapters disclose marker omission in
`x-experiential-ignored-parameters`; a provider label or cached price does not prove support.
`maximize_cache` prefers marker-preserving adapters but may fall back to another provider or
an adapter that drops markers; it is not a cache-capability requirement or hit guarantee.
Cache markers retain their requested TTL. Gateway reservations require the corresponding
five-minute or one-hour price; settlement uses provider-observed TTL counts, never the request
marker. Missing rates or incomplete TTL evidence remain unknown rather than using another rate.
Responses expose observed cache reads and writes; absent write counts remain unknown.
The decoder also
carries the provider-native tool annotations (`strict`,
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
carriers, server tools replay only on the Anthropic wire. A MIXED route (an Anthropic rung beside
another) rejects them by name; a route with NO Anthropic rung serves the turn without them,
dropping the declared tool (`tools.web_search->dropped(unsupported_by_provider)`), its echoed
`server_tool_use` / `*_tool_result` history blocks
(`messages.server_tool_blocks->dropped(unsupported_by_provider)`), the citations of a cited
answer whose text stays (`messages.content.citations->dropped(unsupported_by_provider)`), and a
selector that named the tool (`tool_choice->dropped(unsupported_by_provider)`), because Claude
Code recovers on its own once WebSearch is simply absent while a 400 kills the turn (two
production turns on 2026-09-11). The terminal `message_delta` usage
report supersedes the `message_start` input legs when present, because server-tool turns re-read
fetched results as input and the start-frame count severely undercounts the billed total.

Gateway-executed web search is documented in [gateway-web-search.md](gateway-web-search.md);
gateway-executed tool search in [gateway-tool-search.md](gateway-tool-search.md).

OpenAI-family prefix caches are keyed per cache node behind the provider's load balancer, so
an identical prompt hits but the same stem with a new tail (every turn of an agent loop) is
routed by the whole prompt and usually misses (Tencent TokenHub, measured 2026-09-05: 2 of 8
shared-stem turns hit with no hint, 10 of 10 with one). The gateway therefore dispatches a
`prompt_cache_key` on rungs whose wire profile says the provider routes by it (OpenAI, Tencent
TokenHub, and OpenRouter, whose documented sticky routing falls back to that field as the session
key and forwards it upstream, so a provider's own node pin rides along; other OpenAI-compatible
servers may reject unknown fields, so they never receive it, BYOK or not, and a vLLM origin
ignores the field because its prefix cache is per engine process and content-addressed): never
the caller's raw value, which shares a house account across tenants, but a
digest namespaced by organization and identity (`exp/runtime/gateway/prompt_cache_affinity.py`).
A caller `prompt_cache_key` is the material when present; otherwise the conversation stem (the
leading system/developer messages, which every turn of a session and every request sharing that
system prompt repeat verbatim; the first user turn when there is no system prompt) stands in, so
a Terminus-style loop is pinned to the node holding its cached stem for its whole session with
no client change (measured through the gateway on a hot stem: 9/10 hits keyed vs 4/10 unkeyed). The derived key is dispatch state on the provider request only;
the public request, its digests, and replay identity never carry it. LiteLLM message dumps
(`provider_specific_fields`, null `thinking_blocks` / `reasoning_items` / `images`) decode when
echoed back verbatim: the object is dropped with a `messages.provider_specific_fields`
disclosure and the empty forms are accepted like the SDK's own empty keys, while populated
carriers stay rejected by name.
Anthropic-signed thinking replays only on the Anthropic wire: a mixed waterfall's Anthropic rung
re-emits the caller's blocks verbatim, while every foreign wire omits them at encoding and the
route discloses `messages.thinking->dropped(unsupported_by_provider)` instead of rejecting (the
blocks are baked into a framework-managed transcript, so a session that switches from a Claude
model to any other keeps serving). The Messages surface also carries the gateway's OWN preserved
thinking, mirroring the Chat surface's `reasoning_content` contract: an exposure-gated rung's
(`reasoning_output_exposed`) plaintext reasoning streams and aggregates as one UNSIGNED `thinking`
block (Anthropic signs every block it issues, so an unsigned block is recognizably the gateway's),
and a tool turn's hidden reasoning leaves only as the sealed carrier, in one trailing
`redacted_thinking` block (the carrier is known once every tool call completed, after the
sequential thinking block closed; `redacted_thinking` is Anthropic's opaque replay-verbatim
shape). On replay the decoder maps an unsigned block to the caller-owned plaintext an exposing
rung forwards (dropped with disclosure elsewhere) and a carrier-prefixed `redacted_thinking`
payload to the sealed carrier that admission authenticates and pins to its issuing rung,
dropping the unsigned display duplicate beside it. On the Responses surface over Anthropic routes,
thinking text is projected onto the reasoning-summary channel (signatures deliberately dropped)
so callers receive the reasoning they pay for, while the Chat surface has no reasoning
representation and drops it like summary deltas. Streaming emits the Anthropic
lifecycle (`message_start`, `ping`, content blocks, `message_delta` with the mapped stop reason
and usage, `message_stop`, or one terminal `error` event); the non-streaming body is the
Anthropic message object. Completed streams stop with `end_turn` (`tool_use` when tool calls are
present), token-limited streams with `max_tokens`, and a caller stop sequence that the gateway
matched with `stop_sequence` plus the exact matched string. The Anthropic protocol defines no
idempotency header, so this surface never joins the keyed replay stores.

**Gateway-emulated stop sequences.** The OpenAI Responses API has no stop field, so a rung on
that dialect admits `stop` / `stop_sequences` regardless of its catalog flag and the admitted
route entry carries the caller's exact sequences instead of the payload. The native data plane
cuts visible text at the first match (withholding only the shortest tail that could still start
a sequence, so a match may span delta boundaries), discards what the model says afterwards,
keeps draining lifecycle and usage events so settlement stays exact, and terminates the stream
with a stop-sequence outcome: `finish_reason: stop` on Chat, `status: completed` on Responses,
and `stop_reason: stop_sequence` on Messages. Reasoning, tool arguments, and refusals are never
inspected. Rungs whose provider honours `stop` natively (Chat-compatible, Anthropic, Gemini,
Bedrock) keep forwarding it on the wire.

**Provider-declared stream errors are classified by what the provider said.** A provider that
opens the stream and then declares its own error inside a frame (OpenAI `error` /
`response.failed`, Anthropic `error`, Gemini's error envelope, an OpenAI-compatible `error`
object) no longer collapses to one `provider_internal` 502. The raw code and message classify
it: content verdicts (content filtering, safety, data inspection) are `refusal`; caller-input
phrasing ("exceeds the context window", "does not support max tokens", "invalid params") or a
4xx code is `invalid_request`, a 400 that relays the provider's sentence and never redials or
fails over; rate limits and overloads are `throttled` with `Retry-After`; provider quota,
credential, and model-not-found codes take their HTTP-status classes; only a genuine provider
fault stays `provider stream failed`. An aggregator's 502 wrapping an upstream 400 is read by its
sentence. The bounded ledger detail exempts the request's own model id from the identifier screen,
so a provider sentence naming the model is kept rather than dropped.

**A refusal names its bounded category to the caller.** A `refusal` answer stays a 400 with code
`refusal` and type `invalid_request_error`, but it now carries a machine-readable `refusal_reason`
field in the error body (present on every surface that renders the public error: `/v1/chat/completions`,
`/v1/responses`, and `/v1/messages`, whose Anthropic envelope carries the same field). The category
is a closed vocabulary derived from the provider's own code and sentence, never its prose:
`cyber_policy`, `cbrn`, `content_policy`, `recitation`, `data_inspection`, and `unspecified` for a
refusal the provider filed under no reason. The caller-facing message is the fixed sentence plus the
category's fixed phrase ("provider refused the request: cybersecurity policy"), while the raw provider
token keeps riding `provider_detail` into the ledger only. The reason also rides the settlement
argument next to `provider_detail`, so the control plane counts refusals by reason without parsing the
free-form detail.

**Customer-managed credentials fail as the customer's error.** On a BYOK rung, a provider 401/403
or 402, at stream open or declared mid-stream, is the customer's configuration, not operator
deadness. The failure keeps its ladder class so any other customer-managed rung with its own
credential may still serve, but a terminal answer is the customer's 400
(`provider_credential_rejected` / `provider_account_quota`) naming their provider and what to
fix, and settlement files it as `invalid_request`. House rungs keep the operator-actionable classes.

**Tool calls cut off at the output budget are incomplete, not malformed.** On wires that reveal the
stop reason only after the tool block closes (Anthropic `message_delta`, Bedrock `messageStop`), a
tool call whose arguments END mid-value at its block stop is the provider's cut whatever stop
reason follows (Bedrock's DeepSeek and Qwen shims report `tool_use`): it is dropped and the stream
ends `incomplete` (the caller's remedy is a larger budget). A call whose arguments carry a syntax
error is held rather than failed; a provider-declared `max_tokens` truncation forgives it, while
any other ending surfaces the parse failure as the malformed stream it is, exactly as the
Chat-compatible `finish_reason: length` path already did.

**Pre-stream 4xx bodies keep the provider's code.** When a client-error body's sentence must be
dropped by the identifier screen, the provider's documented code or type token (`invalid_value`,
`INVALID_ARGUMENT`) is relayed instead of nothing, and a content-filter code under a 4xx (Azure,
Gemini) is filed and answered as a `refusal` rather than a request-shape error.

**A reseller's reason token classifies the failure, and a 429 body is read for its token.** On the
OpenAI-compatible dialect Novita's flat envelope (`{code, reason, message, metadata}`, read by the
shared envelope reader) also decides the class by its `reason`: `INVALID_REQUEST_BODY` is a generic
token (classified, never relayed alone), `MODEL_NOT_FOUND` under any 4xx takes the lane policy,
`NOT_ENOUGH_BALANCE` under a 403 is `provider_quota` (the class the house exhaustion sweep reads, not a
credential verdict), `RATE_LIMIT_EXCEEDED` / `TOKEN_LIMIT_EXCEEDED` throttle, and `FAILED_TO_AUTH` /
`ACCESS_DENY` authenticate, pre-stream and inside a stream frame alike. A 429 body is now read too,
under a 250 ms budget so throttle failover stays near-immediate: OpenAI's `insufficient_quota` under a
429 is `provider_quota`, and any other non-generic token rides into the ledger as `http 429: <token>`
(Novita's 429s carry rate-limit headers showing the account's quota untouched, so only the token says
which window closed); the public throttle error is unchanged. A frame with a non-array `choices` stays
malformed and its reason names the frame's sorted key names (never a value).

**Sampling controls a route cannot carry are dropped with disclosure, not refused.** A
`temperature` or `top_p` sent to a route where some rung's provider rejects the field outright (a
reasoning model such as GPT-6 Astra) is dropped and disclosed (`temperature->dropped(unsupported_by_provider)`)
so the model still answers with its own default; the 400 remains only for a value outside a
supporting route's declared range, which is a genuine caller error.

**An Anthropic-shaped `thinking` object on Chat retains its requested depth.** Bare
`enabled` and `adaptive` select the route's default effort with
`thinking->translated(reasoning_effort)` disclosure; `disabled` selects `reasoning_effort: none`.
An enabled `budget_tokens` instead travels unchanged on qualified numeric-budget rungs,
without an effort approximation. The budget must be an integer of at least 1024 and below
the output ceiling. Adaptive/off modes and competing effort or budget controls are rejected.
Top-level Chat `thinking_budget` also exposes Gemini zero/dynamic controls. Unsupported routes return an actionable
`thinking.budget_tokens` error before dispatch. See [generation controls](gateway-generation.md).

**`parallel_tool_calls` is honoured on every route.** A rung whose wire carries the control forwards it.
On a rung without it (Gemini, Bedrock, an OpenAI-compatible server that ignores the field), `true` is
dropped as the provider's own default (`parallel_tool_calls->dropped(provider_default)`) and `false` is
emulated by the data plane, which serializes that rung's stream to its first tool call per turn and
drops later calls in the same turn, start to completion, including their Responses item lifecycle
(`parallel_tool_calls->emulated(serialized_by_gateway)`). The model receives one result on the next
turn and re-issues the remaining calls then, which is the sequential behaviour the caller asked for.

**Pre-dispatch context-window refusal.** Before any reservation or provider call, admission
lower-bounds the prompt's token count from its UTF-8 text bytes (at six bytes per token, below
what real tokenizers produce on prose, code, or CJK text; inline media is not counted) and
refuses with `code: context_length_exceeded` and the exact numbers when even that lower bound
exceeds the largest declared context window on the route. Anything under the bound dispatches
and is left to the provider's precise count; output budgets are never refused here, a too-small
ceiling is an `incomplete` answer.

Exposure-gated reasoning rungs (Tencent Hunyuan and DeepSeek, rows stamped
`reasoning_output_exposed`) accept caller-owned plaintext `reasoning_content` on assistant
history, including tool-call turns. A rung becomes a preserved-thinking carrier route either by
host recognition (Tencent's two OpenAI-compatible origins) or by declaring the rung capability
`reasoning_content_native`, which says the origin returns the standard `reasoning_content` field
and accepts it back (a self-hosted vLLM origin started with a reasoning parser). It is off by
default and fails closed: an undeclared origin has no carrier route, and exposure still requires
`reasoning_output_exposed`. The `prompt_cache_key` node pin stays keyed on Tencent's hosts, since the
declaration says nothing about whether an origin tolerates unknown request fields. The decoder preserves the text verbatim, including an
explicitly empty string: a provider can require the field even when the turn performed no
reasoning. Missing or null values remain absent. Plaintext is bounded to 8,388,608 characters;
values exceeding that limit receive a named error with the limit and a retry instruction.
Route narrowing prefers exposing rungs and discloses
`messages.reasoning_content->dropped(unsupported_by_provider)` when a rung cannot replay it,
including routes with no exposing rung.

A rung whose chat template accepts a system message only as the very first message declares
`system_messages_leading_only` (the official Qwen3.6+ `chat_template.jinja` raises
`System message must be at the beginning.` for any system turn that is not the first message, a
second leading system turn included, so a vLLM origin serving it 400s the whole request; coding
agents inject a system turn after the first user turn and after every tool result). On a
declared rung the Chat wire builder merges a run of leading instruction turns into one system
message and folds every later plain-text system or developer turn into user text in place
(appended to a preceding text-only user turn, else re-roled as a user turn), for the Chat,
Responses, and Messages surfaces alike, and admission discloses
`messages.system->folded(system_messages_leading_only)` when a turn moved. An undeclared rung's
messages are never rewritten; the 400 stays classified as a lane limitation the ladder fails
over.

The instruction-hoisting wires (Gemini `systemInstruction`, Bedrock Converse `system`) carry
instructions only outside the turn list, so a system turn after conversation start has no
positional carrier there. Those rungs fold it the way the Anthropic wire does: the leading
instruction run is hoisted as-is (one part per message), every later plain-text system or
developer turn rides as user text at its position (appended to a preceding text-only user turn,
else its own user turn; Converse merges adjacent user blocks), and admission discloses
`messages.system->folded(system_instruction_wire)`. Until 0.7.77 such a route was refused
outright ("A system message after conversation start is not supported by this model route";
1,747 requests in the seven days to 2026-09-15, almost all Claude Code on `/v1/chat/completions`
against Gemini aliases and Claude aliases whose waterfall carries a Bedrock rung). Gateway-issued carriers are recognized by their
scheme prefix and retain strict parsing, authentication, and route binding; malformed carriers
never become plaintext history, and a carrier never reaches a rung other than the one that sealed
it (the failover past a failed issuing rung strips it, see the reasoning-continuation ladder above).

DeepSeek's own origin (`https://api.deepseek.com`, `is_deepseek_base_url`) is a reasoning-HISTORY
route by origin, independent of the exposure stamp (`GatewayWireProfile.deepseek_reasoning_history`).
Its thinking mode, on by default, rejects a request that carries `tools` unless every assistant
message of the current turn (after the last user message, text-only messages that precede a tool
call included) carries `reasoning_content` (HTTP 400 ``The `reasoning_content` in the thinking
mode must be passed back to the API.``), while accepting an empty string exactly like real
reasoning anywhere, exempt messages included (verified live 2026-09-10). On that rung the Chat
builder forwards caller plaintext `reasoning_content` verbatim on plain and tool-call turns alike,
and every assistant message that arrives without the field or with an explicit `null` (a history
started on another provider, or an OpenAI-compatible SDK that strips the extension) is backfilled
with `reasoning_content: ""` — tool-call and text-only messages alike, since backfilling only
tool-call turns left the "text message, then tool-call message" agent shape 400ing in production;
no other origin is touched. The rung counts as a carrying
rung for narrowing and disclosure (`replays_plaintext_reasoning`), so no drop is disclosed there.
`reasoning_output_exposed` keeps its one meaning on DeepSeek: whether the caller SEES the reasoning
deltas on output. Unstamped, the caller never receives reasoning to replay and the empty backfill
is what keeps agent loops alive; stamped, the caller replays the real text and the backfill only
covers turns minted elsewhere.

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
dropped value reaches the provider through no channel), `strict: true` tools degrade to
best-effort schemas, and a forced `tool_choice` (`required`/`any`, or a named tool) relaxes to
`auto` as `tool_choice->auto`. Two Anthropic wire facts feed those last two
(`exp/runtime/models/providers/anthropic_tool_compat.py`, verified live 2026-09-05): Claude Fable
5.1 and Mythos 5.1 answer a forced choice with a 400 by name on every request, and every model
rejects a forced choice beside a budgeted `thinking: enabled` config, so the Anthropic builder
declines those requests as `forced_tool_choice` before dispatch (narrowing prefers a rung that can
force a tool, such as an aggregator rung of the same alias, and keeps the caller's selector
verbatim there); and the strict validator compiles tool schemas into a grammar and 400s by name on
keywords it cannot express (`maxItems`, `oneOf`, `minimum`, an unsupported `format`, a recursive
`$ref`, ...), so a strict tool using one is declined as `strict_tools` on Anthropic rungs, which
narrows to a strict-capable rung when the route has one and otherwise drops only `strict`, never
a schema keyword. The same validator requires `additionalProperties: false` on every object, so
strict tool schemas reaching an Anthropic rung have their objects closed with the
`tools.parameters.additionalProperties->false` disclosure, exactly like structured-output schemas.
The `capability_parity` row reports the per-release forced-choice fact as
`supports_forced_tool_choice`. On the OpenAI-compatible Chat Completions wire a canonical
`developer` message is emitted as `system` without disclosure: OpenAI defines the two roles
identically (developer-provided instructions the model follows regardless of user messages),
while the third-party servers behind that dialect enumerate only the classic roles and reject
`developer` by name; the native Responses wire keeps the role it defines. The Anthropic wire also rejects an empty text block anywhere and a turn whose text is all whitespace, while it accepts an empty assistant content array in any position (verified live 2026-09-05): an assistant turn with no readable text dispatches as an empty array, a system prompt with none is omitted, empty blocks inside richer turns drop with their cache breakpoints migrated, and an empty or whitespace-only user turn (which no array form can carry) is refused by name before dispatch. On a reasoning route that accepts sampling only at `reasoning_effort=none`
(`sampling_requires_reasoning_none`, e.g. gpt-5.6-sol/luna), a `temperature`/`top_p` sent with
reasoning on is dropped and disclosed as `temperature->dropped(set_reasoning_effort_none)` rather
than rejected — the model accepts sampling, just not at that effort, so the request serves and the
caller is told how to keep the value (set `reasoning_effort=none`); a route that never declares the
control at all (Anthropic constrained `[1,1]` sampling) still hard-rejects it, since there is
nothing to honor at any effort. `top_k` prefers a carrying rung; when no rung supports it,
admission drops it with `top_k->dropped(unsupported_by_provider)` because defaults still serve.
`frequency_penalty` and `presence_penalty` follow their per-rung capability truth and otherwise
drop with `<parameter>->dropped(unsupported_by_provider)`. These are soft preferences.
Chat token probabilities are opt-in on verified compatible rungs; see [Chat token probabilities](chat-logprobs.md).
Native Responses probabilities use their own capability and selector; see [Responses output probabilities](responses-logprobs.md).
A caller
`response_format: {type: "json_object"}` requests schema-free JSON output. OpenAI-compatible
rungs use native JSON mode, Responses rungs use `text.format: {type: "json_object"}`, and
Gemini uses `responseMimeType: "application/json"` without a schema. Anthropic/Bedrock use a
best-effort system instruction, disclosed as `response_format->instruction(json_object)`.
Every wire receives a counted JSON-object instruction; native format fields are retained.
No empty schema is synthesized. Use `json_schema` when a supported route must enforce a shape.
A caller `service_tier` on the OpenAI-family surfaces forwards verbatim
only on rungs dispatching tenant-owned (BYOK) credentials, where the caller pays the provider
directly; host-funded rungs never emit it (the tier changes provider pricing while the gateway
bills catalog rates) and a route with no eligible rung drops it with disclosure. Anthropic's own
`service_tier` stays a recorded Messages-surface rejection. A caller top-level `provider` object (OpenRouter's routing-preference shape) is accepted on all three surfaces, Messages included (Anthropic SDKs send it through `extra_body`); exactly one key changes gateway behavior: `provider: {"zdr": true}` DEMANDS zero-data-retention routing for that request, carried as `GatewayRequest.zdr_requested` and `AuthorizationSnapshot.zdr_requested`. A host that publishes provider data-retention postures applies the same posture filter as its organization-level `require_zdr` (natively ZDR rungs first, then an OpenRouter rung dispatched under `provider: {"zdr": true, "data_collection": "deny"}` plus `X-OpenRouter-Metadata: enabled`, flagged through `ExecutionSnapshot.zdr_constrained_deployment_ids`), answers `x-gateway-zdr: true`, and refuses with a 403 naming the excluded providers when no rung qualifies; the demand only tightens and never loosens an organization policy, and the local gateway (no postures) refuses it with a 403 on `provider.zdr`.
The rest of the object (`data_collection`, `order`, `only`, ...) forwards to OpenRouter rungs verbatim (tightened when the rung is constrained) and is dropped on every other wire (`openai_responses`, `anthropic_messages`, `gemini_generate_content`, `bedrock_converse_stream`, and non-OpenRouter `openai_compatible` rungs), which have no such field. On `maximize_cache` pools, a cache-marked request dispatches
marker-honoring (Anthropic Messages) rungs before marker-dropping wires, stably within each
group, so a shim rung can no longer silently bill every turn's full context uncached while the
native rung stands ready; routes narrowing to only marker-dropping wires keep disclosing the
dropped markers. On `maximize_cache_affinity` pools the certified initial order is replaced per
request by a weighted rendezvous hash of the request's stable conversation identity (the caller's
`prompt_cache_key`, else a Responses continuation's original episode key, else the session-scoped
`X-Client-Request-Id`, else the idempotency key, else the request id) over the pool's rungs, with
weights from each rung's authored `GatewayRungDispatchPolicy.affinity_weight`. Every worker
computes the identical permutation from catalog data alone (no per-worker memory, no shared
state), so one conversation lands on the same rung fleet-wide and, when that rung sheds or dies,
on the same deterministic alternate, building warm cache there instead of scattering; a rung's
death or restoration moves only its own fingerprints. Failover semantics under this mode are
availability-style (a throttle fails over to the deterministic alternate), and the cache-marker
partition above still applies first on marked requests, rendezvous-ordered within each group. Every coercion is disclosed in `path->effective` form
through `ignored_parameters`, logged, and counted in the `admission_parameter_coercions`
metric; every serving surface carries that list to the caller as a body-level
`x-experiential-ignored-parameters` key (Chat chunk and completion, Responses envelope, and the
Anthropic message on both `message_start` and the aggregated body), so a drop is never silent; nothing coercible keeps the first rung's own field-scoped rejection.
When a coercion APPLIES but none SERVES (every candidate dies one layer later, on a blocker
unrelated to the coerced field), admission carries the unprobed coercion forward so the stage
that actually refuses names the caller's remedy: an image on a text-only reasoning route is
refused on `messages`, never as an unsupported `thinking` field that merely rode along.
Thinking disclosures name an applied translation, not consent to drop a cost constraint.
A numeric thinking budget and an explicit off setting must be enforceable on the selected
wire; otherwise admission refuses the named field before dispatch. A budget-less enable can
translate onto an admissible effort ladder with a `thinking->reasoning_effort` disclosure.
A valid native thinking configuration is kept by narrowing the route before considering a
translation. A route that offers reasoning but cannot honor the requested control is not
silently served at its default depth. See [generation controls](gateway-generation.md).

On the Messages stream, `message_start.message.usage` is a PRE-DISPATCH figure and
`message_delta.usage` is the authoritative one. Every Messages usage object (`message_start`,
`message_delta`, and the non-streamed body) carries Anthropic's four legs, always present and
`0` when the provider reported none: `input_tokens` (uncached input), `cache_creation_input_tokens`,
`cache_read_input_tokens`, and `output_tokens`. The cache legs come out of the normalized
folded total the ledger bills: an Anthropic rung's own `cache_read_input_tokens` /
`cache_creation_input_tokens`, an OpenAI-wire rung's `prompt_tokens_details.cached_tokens`
(Chat) or `input_tokens_details.cached_tokens` (Responses), Gemini's `cachedContentTokenCount`,
Bedrock's `cacheReadInputTokens`; Anthropic and Bedrock report cache writes. Other rungs
carry `cache_creation_input_tokens: 0`. The start frame carries what the upstream already
reported before content — an Anthropic upstream's own start-frame input and cache meters,
mirrored — and otherwise the control plane's pre-dispatch count of the prompt (the reservation
estimator without its headroom, carried on the admission as `input_token_estimate`, the same
count `count_tokens` answers) in Anthropic's documented start shape,
`{"input_tokens": N, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 1}`
(nothing is cached before dispatch). The count covers the system blocks, every turn's content
blocks (tool_use and tool_result included), and each tool definition, so a Claude Code session
reads in the thousands there; a two-digit figure is a two-digit prompt (Claude Code's own
startup probes), never a stub. An OpenAI-wire upstream reports nothing before its final chunk,
so without the estimate its `message_start` showed the zero placeholder that clients reading
input from the start frame alone (Claude Code) display as 0 input tokens. `message_delta.usage`
carries the provider's final meters plus the platform's cost extensions; the official Anthropic
SDK accumulators (Python and TypeScript) copy every usage field present on `message_delta`, so
their final message shows the true counts. The estimate is display-only: the encoder keeps it
apart from the usage it settles from, so it never reaches `message_delta` or the ledger, which
bill the provider's report.
The per-deployment `capability_parity` export joins catalog declarations with provider-family ground truth so catalogs can warn about gaps and route around them.
Schema version 7 additionally projects `supports_prompt_cache_boundaries`, `supports_custom_tools`, `supports_grammar_tools`, `supports_tool_call_limit`, `reports_model_status`, and the existing `reports_reasoning_tokens` declaration.
These declarations default false and never enable a request feature. Cache boundaries mean explicit breakpoints and retention, not implicit prefix caching. Read tool flags with the row's `dialect` and public API surface: Chat still refuses custom and grammar tools; Responses still refuses `max_tool_calls`.
Gemini `modelStatus` is not preserved; catalogs must leave `reports_model_status` false until the response contract carries it.

Commit-independent headers are available before streaming begins. Route-dependent headers are
emitted only after an execution snapshot exists. Stable public IDs do not expose raw key,
idempotency, request, or provider values.

`GET /v1/models` lists only the aliases granted to the presented key. The envelope contains
only the OpenAI `object` and `data` fields, and every entry contains only `id`, `object`,
`created`, and `owned_by`. Capability, pricing, revision, and catalog-digest metadata never
ride this compatibility endpoint. Platform's separate `/api/models` catalog owns rich route
metadata, including configured nano-USD-per-million-token prices.
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
encrypted payload to its original item id and callers echo this gateway's own minted public ids.
The provider also binds the payload to the organization (Azure: the tenant) that sealed it, so a replayed item another lane, account, or tool produced is refused with `invalid_encrypted_content` whatever rung dials; the data plane treats that refusal as a repair, not a verdict, whether the refusal arrives before the stream (OpenAI's 4xx) or inside it (OpenRouter's Responses relay answers 200 and fails the stream on its first frame under `invalid_prompt`): the waterfall re-dials the SAME rung once through a new reservation when the [request attempt budgets](gateway-request-policy.md) permit it, with every replayed reasoning item carrying `encrypted_content` stripped (message, tool-call, and tool-result items keep their positions; the calls and assistant message of a stripped turn lose their provider `id`, which the provider would tie back to the missing reasoning item), remembers that stripped payload for every later dial of the rung in the same request (a throttle redial never earns the refusal twice; another rung starts from the original, since its account may decrypt it), and remembers per worker, for 30 minutes after their last replay and up to 100,000 entries, the digests of the refused payloads (the one the provider quoted, else every one present; salted by the caller's organization and identity ids as admission names them, never by the request's bearer, which in a hosted worker is the front's ephemeral exchanged token, so one caller's memory never touches another's) so a LATER request that replays them is stripped of exactly those items before its first dial (`encrypted_reasoning_stripped_proactive`), with the reactive repair still behind it, and discloses a served repair through the `x-gateway-replay-repair: encrypted_reasoning_stripped` header (HTTP responses only), the `encrypted_reasoning_stripped` data-plane counter, and one content-free operator line, each written only once the re-dial opened.
The verdict is read through the shared OpenAI-family envelope reader: OpenAI's and Azure's `error.code`, OpenAI's fixed sentence when a relay such as Novita re-envelopes the refusal without the code, and the document OpenRouter relays under `error.metadata.raw`; hidden reasoning continuity is lost for the stripped items only; without the memory the repair would RECUR on every later turn (the stateless caller keeps the foreign items and a `previous_response_id` continuation retains the turn as replayed; production 2026-09-15: about 74 refused dials a minute, 82% of one tenant's agent turns), so a conversation now pays the refused dial once per worker that serves it; a refusal of the stripped payload itself, or a payload with nothing to strip, surfaces the provider's 400 unchanged. Replay is opt-in through the standard `Idempotency-Key` header only;
`X-Client-Request-Id` is caller correlation identity (Codex sends its session id there on
every request of a session), echoed on responses and used for route affinity, never as an
operation key. Restart
or eviction returns an explicit unavailable error and never reconstructs content from SQLite.

## Content-free observability and lifecycle

The accounting SQLite stores hashes, frozen authority, route identity, state transitions, tokens,
latency and estimated cost, never content or keys. Separate [local capture](local_gateway_traffic.md)
is default-on (`--ghost` disables it). `GET /usage` and `GET /usage.json` render the same schema-v2 report
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

`exp/runtime/gateway/provider_certification.py` is the dated provider matrix, including client,
gateway surface, wire, fixture result, and live status. Its live cells remain
`not_run_requires_credentials`; fixtures alone prove no hosted availability or account behavior.
TypeSafe cells name decisions, not chat: real Rust HTTP plus SQLite tests use a loopback provider.
A direct TypeSafe API smoke with synthetic input exercised all three question types on 2026-09-16.
That direct success is not hosted-gateway, deployed-fleet, price-invoice, or reliability certification.

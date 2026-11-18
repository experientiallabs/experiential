# Per-rung conditional failover (`failover_only_on`)

A deployment may carry
`GatewayDeploymentCapabilities.failover_only_on`, a set of failure tokens: the failover-eligible
failure classes by wire name (`throttled`, `timeout`, `transport`, `provider_internal`,
`provider_quota`, `provider_authentication`, `provider_not_found`, `unavailable`, `empty_completion`,
`malformed_response`, `guardrail`), `refusal` for any provider refusal, or `refusal:<reason>` for
one bounded category (`cyber_policy`, `cbrn`, `content_policy`, `recitation`, `data_inspection`,
`unspecified`; the closed list is `exp.common.models.failover_tokens.FAILOVER_TOKENS`, mirrored by
the data plane's `waterfall/fallback_rules.rs`). Such a rung is FAILOVER-ONLY: it is never chosen
for a request's first dial (nor reached by a sideways shed or a budget skip) and is claimed as a
successor only when the failure the ladder is walking from spells one of its tokens — and then even
for a refusal on an alias revision that did not enable refusal failover, because the rung's own
opt-in is the caller's remedy (a customer's OpenAI key enrolled in a trusted-access program taking
over exactly the requests the house rung refused under `refusal:cyber_policy`). A rung with no set is
unrestricted and behaves exactly as before; unrestricted rungs keep their order and the route policy
still governs advancing onto them. Refusal DELTAS carry no reason, so a rule rung downstream causes
them to be withheld only when its set accepts an unnamed refusal (`refusal` or `refusal:unspecified`).
The wire fact rides each route entry as `failover_only_on` (null when unrestricted); the control
plane chooses the depth (`native_fallback_rules.py`) and records the reason on the attempt as
`fallback_reason = failover_only_on:<token>` (the failure's precise token); the data plane decides
whether a successor exists before asking, and refuses a reservation that violates the rules (a first
dial or an unmatched successor on a rule rung) with an internal error rather than dialing it. A route
whose EVERY admitted rung carries a set fails closed at admission with a named internal error
(`FailoverRulesError`), never an exhausted ladder that dialed nothing.

## Authoring

Set `failover_only_on` on the deployment's `GatewayDeploymentCapabilities` (the platform projects it
from the lane's dispatch facts, beside `time_to_first_byte_base_seconds`). Tokens are validated
against the closed `FailoverToken` literal, so a misspelled or never-failover class (`invalid_request`,
`quota_exceeded`) is refused at the typed boundary. An empty set is a rung that is never dialed at all.

| Set | First dial | Successor to `refusal:cyber_policy` | Successor to `throttled` |
|---|---|---|---|
| (none) | yes | only with the alias revision's refusal failover | yes (policy) |
| `["refusal:cyber_policy"]` | never | yes, regardless of the revision's opt-in | never |
| `["refusal"]` | never | yes | never |
| `["throttled", "provider_internal"]` | never | never | yes |

## Ledger

An attempt reserved on a rule rung as a successor carries `fallback_reason =
failover_only_on:<token>`, where the token is the FAILURE's precise spelling (`refusal:cyber_policy`
even when the rung's set is the bare `refusal`). First dials, same-rung redials, and unrestricted
rungs keep the route's own `fallback_reason`.

## Failure modes

- Every admitted rung restricted: admission raises `FailoverRulesError`, answered as the gateway's
  internal error and logged with the exception type, before any reservation.
- The control plane reserves a rule rung for a first dial or for a failure its set does not name:
  the data plane settles the attempt `failed` / `internal` ("failover-only rung reserved outside
  its rules") and answers 500 without dialing the rung.
- A rung restricted to one refusal category never takes an unnamed refusal (`refusal:unspecified`),
  and refusal text streamed by the provider is withheld for a rule rung downstream only when that
  rung accepts unnamed refusals.

A rung at its per-worker in-flight bound is a different kind of bypass — a policy shed, not a
failure — and what happens when every rung is shed is described in
[gateway-lane-saturation.md](gateway-lane-saturation.md).

## Graceful connection retirement

An open-phase, typed remote HTTP/2 `GOAWAY/NO_ERROR` is a graceful connection
retirement for health observations: it releases a probe without adding a circuit
failure or clearing earlier failures. Explicit retries retain their normal attempt
limits and separate reservations. This health hint does not prove that the request
was unprocessed; unknown usage and its accounting hold remain unresolved. Ordinary
transport failures, midstream failures and `REFUSED_STREAM` keep their existing
health behavior.

## Transport, buffered answers and context windows

A `transport` failure (a reset or refused connection) is the path to the provider failing, not its
answer, so it is never decided by cache stakes: under every failover mode it advances to an untried
sibling before any redial of the failed rung, taking the cache miss, and redials that rung only when
no sibling can take it. When a request's only dial failed and the only remaining way to an answer is a redial of that rung
while this worker holds its circuit open, the redial is admitted through the last-resort and forced
claims a fresh request's first dial would get (a provider-stated throttle window still refuses), so
the circuit never ends a request after one dial; any later redial stays behind it.

Commitment is the moment the caller could see the rung. A surface that assembles the whole answer
before sending it (every non-streaming request, and a stream an output guardrail buffers) drains the
committed attempt inside the waterfall (`WaterfallContext::buffered`, `waterfall/buffered.rs`): a
`transport`, `timeout`, `provider_internal`, or `malformed_response` failure the classifier marked
redialable or failover-eligible that breaks such an attempt mid-body settles it `failed` (its own
attempt row, with whatever usage it reported) and takes the ladder like a pre-commit failure,
because the caller has received nothing. A stream that already sent the caller a delta never
restarts or splices another rung: it ends with the protocol's own terminal error (a Chat `error`
data frame then `[DONE]`, Messages `event: error`, Responses `response.failed`), never a silently
cut connection.

A provider context-window refusal (`context_length_exceeded`, or a sentence such as vLLM's "maximum
context length is N tokens") keeps the caller's `invalid_request` class and sentence, but a lane can
serve less than the window the model advertises. The data plane files it with
`context_window_rejected` (and the stated window in `context_window_limit_tokens`), never redials
it, and keeps it failover-eligible once per request; the control plane advances only to an untried
rung whose advertised window exceeds the stated one (else the failed rung's advertised window), or
whose window the catalog does not state (`native_context_window.py`). With no such rung, or on a
second refusal, the caller receives the provider's 400 unchanged.

## Malformed tool calls

A model can sample a client tool call whose streamed arguments never become one JSON object
("streamed tool arguments are not valid JSON"); another dial, on a sibling lane or the same one, will
usually sample a valid call. So a turn whose first outward output is a client function call is held
uncommitted until the turn ends (`waterfall/commit.rs`, `HeldToolCalls`): the call's opening, its
argument deltas, and every later event wait in order, and the turn commits on its successful
terminal with exactly the events the caller would otherwise have received live. The hold is bounded
(256 KiB, each held event charged its full owned size, or 8,192 events): past it, or as soon as the provider reports a tool it
executes itself, the turn commits and streams live as before. A freeform (custom) tool is not held:
native custom input never fails to parse, and a translated wrapper keeps its never-replay contract. Output the caller already saw (text
before the call, a provider-executed tool) is never held.

A held call that fails to validate is filed `malformed_response` with `malformed_tool_call`, and
the waterfall retries it once per request, taking the cache miss: the control plane prefers an
untried sibling (like a `transport` failure, under every failover mode) and redials the same rung
only when no sibling can take it, with no backoff and under a fresh `Idempotency-Key` (a resample is a
new provider intent, never a replay). The retry runs under the request deadline, not
the pre-first-byte budget the held turn has already been generating in. Settlement carries the flag
to the failed attempt row. A second malformed call in the same request is its answer. A buffered
surface (non-streaming, or a guardrail-buffered stream) follows the same one-retry rule. A malformed
call in a turn that already streamed to the caller keeps the protocol's terminal error event.

## Provider content filters

A provider's own content filter (Azure's content management system, an account-side moderation
layer) can refuse a request, or the output it would have returned, that another provider serving the
same model applies no filter to. The data plane files such a verdict as a `refusal` with
`refusal_reason = content_policy` and `provider_content_filter` (`content_filter.rs`) when it comes
from the provider's typed rejection: a 4xx whose completion body finished `content_filter`
(Azure Foundry's `content_filter_results` verdict), a 4xx error envelope whose code is a
content-policy code (directly or relayed inside a decode-failure 400), the relayed Gemini
`PROHIBITED_CONTENT` 403, an in-stream error event carrying a content-policy code, a Chat stream
closed with a `content_filter`/`safety` finish, a Responses stream ending `incomplete` for
`content_filter`/`safety`, or a Bedrock `content_filtered`/`guardrail_intervened` stop. It is never
set for the model declining in its own words (a Chat `refusal` delta, a refusal stop reason), for a
refusal read only from a sentence, for a cyber, CBRN, recitation or data-inspection verdict, or for
the gateway's own `guardrail` class. Settlement carries the flag, so the failed attempt row tells
the provider's filter from the model's refusal.

Such a refusal is never redialed, and the waterfall makes it failover-eligible once per request,
only when an untried unrestricted rung of a different provider exists and only while nothing reached
the caller: before commit, or while a buffered surface (non-streaming, or a guardrail-buffered stream)
drains a committed attempt. The empty refusal marker an OpenAI-compatible filter finish emits is held
uncommitted while some rung may still take the refusal (the one-shot gate, a `failover_only_on` rung
naming it, or the alias revision's refusal failover). Only a content-policy CODE marks a filter: a
generic `refusal` code whose category comes from the sentence does not. The control plane then admits only an untried rung whose provider
differs from the refusing rung's (`native_content_filter.py`); another deployment of the same provider
would apply the same filter. The refused dial keeps its own failed attempt row, and the served attempt
is recorded at its own route depth like any failover, taking the cache miss. With no such rung, on a
second filter refusal, or once a live stream sent the caller a delta, the caller receives the refusal
exactly as before. An alias revision that opted into refusal failover keeps its own unrestricted walk,
and a `failover_only_on` rung that names `refusal` or `refusal:content_policy` keeps taking such a
refusal on its own opt-in, as it did before this rule (the one-shot gate governs only unrestricted rungs).
The walk never leaves the request's route, so it never changes the model.

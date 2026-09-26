# Generation limits and interrupted streams

The gateway preserves caller intent across provider protocols. An omitted optional field is
not permission to select a smaller output budget or a more expensive reasoning mode.

## Output limits

Chat Completions accepts `max_completion_tokens`, `max_tokens`, or `max_output_tokens`
(one non-null spelling per request) through its protocol decoder. Messages requires `max_tokens`. An explicit limit remains a ceiling:
a route that requires a larger minimum is excluded or refused before dispatch, not served
with a silently increased limit.

When the caller omits a limit:

- Optional upstream fields remain absent, including Gemini `maxOutputTokens`.
- Anthropic Messages requires a numeric limit. Its adapter derives that value separately
  for each candidate from its declared maximum output, bounded further by its total context
  window when present. A context window alone does not prove a legal output parameter;
  without a declared output maximum the caller must supply a cap. The adapter discloses
  the required-field default through `x-experiential-ignored-parameters`. This is an adapter
  decision, not a numeric default promised by Anthropic or OpenRouter.
- The public request retains the distinction between omission and an explicit value.
- Accounting freezes a finite reservation bound beside each candidate's actual payload.
  An optional wire can omit its limit while reserving against a declared model ceiling.
  If neither a caller limit nor a declared bound exists, admission asks for an explicit limit
  instead of inventing one.

A total context window is not an exact remaining output allowance. The adapter does not
subtract heuristic prompt-token estimates and call the result exact. Providers still validate
combined input and output limits. Explicitly budget a coding session and compact its context
before it exhausts the model's input allowance.

The selected ceiling includes all output the provider counts, which may include reasoning.
It does not promise that every billed output token appears as visible text or tool arguments.

## Responses input lifecycle markers

Responses input messages with role `user`, `system`, or `developer` may include
`status` without an output item ID, as in the official input-message contract.
Their status does not create assistant replay identity. An assistant output
message carrying `status` still requires its item ID; replay phase markers keep
their existing identity requirement.

## Reasoning controls

Valid explicit thinking-off settings remain off. An unsupported off setting is a typed
pre-dispatch refusal, not a request to use the model's default thinking behavior. Model
families distinguish support for budgeted thinking from support for disabling thinking.
Claude Opus 5.5 always uses adaptive thinking, so an explicit `thinking.type: disabled`
is rejected before dispatch at every effort level. Omit thinking or explicitly use adaptive
thinking and choose an effort level instead. Opus 5 and older releases keep their own rules.
Opus 5.5 also cannot force a tool selection: the existing capability policy uses `auto`
only after all eligible routes decline the forced choice and discloses `tool_choice->auto`.
This does not guarantee a tool call. See the provider's
[Opus 5.5 contract](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide).

An explicit effort cannot be coerced upward. If a route needs a supported spelling, only an
admissible lower tier may be selected, with disclosure; a reasoning-capable route with no
admissible setting is refused. A genuinely non-reasoning route retains its existing disclosed
omission policy. Omitted thinking and effort remain omitted unless the provider protocol requires
an authored setting. A model name appearing inside a system message is not an API setting.

A caller that explicitly requests budgeted thinking without a budget receives a derived
budget that must fit the effective per-provider output ceiling. An explicit budget is not
rewritten to fit. An impossible combination is refused before provider dispatch. Numeric
Messages budgets are forwarded only to budget-capable wires, never converted into advisory
effort or adaptive thinking.

The Chat field `thinking_budget` preserves the caller's numeric control on qualified
Anthropic, Gemini 2.5, and native Qwen Cloud routes. Use it as an OpenAI SDK `extra_body`
field (or a top-level field in HTTP JSON). It is not combined with `reasoning_effort`.
The gateway translates the field name for the selected provider and discloses that
translation; it does not replace the number with an effort tier.

| Native provider route | Models | Provider control and constraints |
| --- | --- | --- |
| Anthropic Messages | Thinking-capable Claude 3.7, Sonnet/Opus 4 through 4.6, Haiku 4.5, Mythos Preview | `thinking: {type: "enabled", budget_tokens: N}`; integer >=1024 and below total output limit |
| Gemini generateContent (including native Vertex Gemini) | Gemini 2.5 Pro | `generationConfig.thinkingConfig.thinkingBudget`; 128..32768, or -1 dynamic; zero is refused |
| Gemini generateContent | Gemini 2.5 Flash / Flash-Lite and qualified previews | Same field; Flash 0..24576; Lite 512..24576; both also accept 0 off and -1 dynamic |
| Native Qwen Cloud Chat | Qualified Qwen3/3.5/3.6/3.7/3.8 and Qwen3-VL identities; GLM 4.7/5/5.1/5.2; Kimi K2 Thinking/2.5/2.6/2.7 Code | `thinking_budget` plus the model's enable switch; nonnegative integer |

Only explicitly qualified model identities and dated snapshots are admitted, not arbitrary
future families or all models sharing a brand. Qwen Cloud uses its native HTTPS endpoints;
other compatible hosts do not inherit this contract. Kimi K3 and GLM 5.3 are excluded because
they do not honor this parameter. Gemini 3 numeric-budget compatibility is not adopted;
use its effort control. A zero Chat budget must be sent without additional on/off controls
because zero's meaning is provider-specific. `-1` is qualified only on Gemini 2.5.

Chat also accepts `thinking: {type: "enabled", budget_tokens: N}` with integer N >=1024;
Messages clients use the same nested shape. The exact number travels to any qualified
provider using the fields above. Nested budgets require room below the caller's output
ceiling. Off/adaptive modes, competing budgets/efforts and unsupported additional thinking
controls fail explicitly. Budget values participate in replay identity, including zero and -1.

The gateway's output limit and reservation cover thinking plus the final answer. Qwen Cloud
models documenting `max_completion_tokens` receive the combined cap there. On models where
`max_tokens` limits only the answer when a thinking budget is supplied, the gateway sends
`max_tokens = total output limit - thinking_budget` and discloses that translation. A budget
leaving no answer room is refused. Omitted output limits are filled from the selected rung's
finite declared bound before payload freezing. Each fallback retains its own bound.

Provider budgets are targets or provider-defined limits, not a gateway guarantee of the exact
observed reasoning length. The gateway preserves the requested control and separately binds
the provider's output ceiling to its reservation.

Sources: [Anthropic extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking),
[Qwen Cloud Chat](https://docs.qwencloud.com/api-reference/chat/openai-chat),
[Gemini generateContent thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking).

OpenRouter's `reasoning.max_tokens` remains unsupported on Chat. Remove that numeric
budget and select an effort, or use the supported numeric controls above on a qualified route.

## Truncated tools

Streaming tool argument fragments retain their order and bytes. Missing arguments, an empty
string, an explicit `{}`, and incomplete JSON are distinct facts.

A provider-declared output-limit termination remains `finish_reason: "length"` on Chat.
Missing arguments on that termination do not become a synthetic `{}` or a completed tool call.
Anthropic and Bedrock block-stop events do not establish a successful final stop reason;
missing-argument completion waits for the terminal reason. Nonempty valid calls can still
complete incrementally. Ordinary complete zero-argument calls remain supported.

A non-streaming aggregate omits incomplete tool calls and retains the incomplete outcome.
Clients must not execute a partial tool call from a length-limited answer. The gateway does
not silently repair arguments, regenerate the answer, or retry a length termination.

A provider's normal completed call containing `{}` is passed through. Completion is not a
promise that the call satisfies every caller-defined JSON Schema constraint or that a client
successfully executed it. Schema validity, provider outcome, delivery, and billing are separate
facts.

## Connection loss

Unkeyed Chat and Messages streams observe receiver closure while awaiting the same provider
read. Closing the response drops the upstream transport before waiting for settlement.
Periodic SSE comments keep a publicly silent stream active without restarting provider reads,
changing first-token measurements, or extending provider/request deadlines.

Closing a transport is not proof that every provider stops its own compute immediately.
Observed usage remains billable according to the host's policy. Cancellation must never erase
usage already parsed or override a provider terminal already observed. The settlement guard
retains these facts across cancellation of the owning task and delivers one decided outcome.

Keyed Chat owners retain their bounded replay contract after their subscriber leaves. A retry
joins or retrieves the same operation rather than dispatching a second generation. Heartbeat
comments are not retained as replay content. Requests without an operation key are distinct
submissions even when their prompts match; the gateway does not deduplicate unrelated callers
by transcript contents.

Some providers supply usage only at the end. A dispatched request canceled before a provider
terminal is observed carries the internal `usage_incomplete_due_to_disconnect` accounting signal
even when partial counts are known, together with the generated text observed so far
(`streamed_output`: visible text and tool arguments in one leg, reasoning in the other, bounded
with an overflow character count). When the provider had accepted the request (`opened`), the
accounting registry completes the meter with the gateway's own tokenizer: the counted prompt fills
a missing input total, the observed deltas fill a missing output total (reasoning folded in as an
output subset), an observed leg is kept when it is at least the estimate, an unreported cache-read leg is
estimated at the organization's recent cached share of input on that rung (the same settled-meter
EWMA the cache-priority term reads; zero without a live sample), and cache-write legs stay unknown
unless the provider reported an input total. The terminal then carries the internal
`usage_estimated` marker and settles at the estimated cost with `usage_source = estimated`,
releasing the rest of the reserved bound; the local gateway's monthly allocation charges the same
figure. A disconnect the provider never answered (not opened), or one on a DECISIONS request,
keeps the conservative policy: the full reserved bound consumes local budget, the provider cost
stays unknown, observed partial counts remain available, and hosted accounting retains unresolved
authorization separately from settled spend. An observed terminal with missing usage retains the
host's unknown-terminal policy; neither marker broadens it. Stopping generation is not evidence
that its unreported usage was free, and an estimate is never reported as observed.

Meter parsing follows the provider's wire contract. Missing primary counts on partial
OpenAI, Anthropic, or Bedrock reports remain unknown. Gemini's present `usageMetadata` uses
implicit-presence protobuf scalars, so omitted zero-valued counts retain Google's zero-default
meaning; thinking tokens still contribute to billed output. An absent usage object is unknown.
Cumulative reports from one generation are merged, not added. Costs from separate physical
generations cannot acquire a known total by adding a known count to an unknown one.

## Gemini usage trailers

Native Gemini and Vertex Gemini routes use upstream `streamGenerateContent` SSE even when
Chat callers request a non-streaming JSON response. A candidate's `finishReason` freezes its
content and outcome, but the gateway continues reading metadata so `usageMetadata` can arrive
before, alongside, or after that finish. Later text, tools, errors, and finish reasons cannot
reopen or replace the declared answer. The final meter precedes one terminal settlement.

This metadata-only drain has one absolute allowance: the selected connection's existing body-read
timeout, measured from the decoded finish and capped by the request's remaining deadline.
Trailers and keepalives cannot renew it. EOF completes immediately, with no fixed waiting period.
A stalled transport can therefore add up to that allowance before the response finishes.
Cancellation closes the upstream without waiting for more metadata. At the hard request deadline,
public SSE delivery can close without a final client frame; settlement still preserves an already
decoded provider outcome and the meter observed so far. This guarantee starts after the finish
passes framing and normalization. The shared SSE decoder rejects an entire network batch on a
framing error, so a finish earlier in that rejected batch is not yet a decoded outcome.

Partial or empty suffixes cannot erase earlier counts. Prompt, candidate, thinking, and cache
counts accumulate independently of the last publishable meter; withholding a cache subset does
not discard output counts that arrive before input. Cumulative snapshots are never added together.
Cache counts greater than accumulated input stay pending, even when a consistent meter already
exists. A later input report promotes the greatest actual observed cache count it covers, while
larger counts remain pending. Smaller newly valid cache reports can therefore advance the meter
independently of a larger pending count. Counts are never clamped to invent a subset. The set of
distinct pending counts is bounded by the existing 4,096-entry provider-state limit; an overflow
ends the drain with the already decoded outcome and last safe meter.

Newer consistent primary counts can advance the meter without publishing pending cache counts.
A report that itself contradicts the subset relation cannot replace the last consistent meter.
Without reconciled input, pending nonzero cache evidence cannot authorize new output-only usage,
even after an empty report; previously observed output legs remain pending until input arrives.
An absent whole usage object stays unknown, and a finished all-zero report keeps the existing
unknown-meter settlement policy. An interrupted or expired drain preserves the best consistent
report, not a guarantee that the provider's final report was received. It does not trigger another
generation to recover missing usage.

This behavior applies to new requests. It does not reconstruct historical provider frames,
attribute past missing meters to a particular cause, or authorize retrospective billing changes.

## Host-authorized Google cache resources

Native Gemini and Vertex can create explicit cache resources only when the embedder supplies
`NativeControlPlane(..., explicit_cache=host)`. The default is `None`, which leaves existing
implicit caching and ignored-marker disclosures unchanged. A host must require both an explicit
five-minute ephemeral checkpoint and a configured customer-funded cache allowance; missing or zero
allowance never authorizes creation. `prompt_cache_key` alone is not a spending instruction.

The initial native path handles exact text prefixes on Google's official Gemini `v1beta` and
Vertex `v1` endpoints. It retains the marked prefix, its system instructions and supported function
tools in one cache resource, and sends the remaining content with `cachedContent`. A checkpoint
cannot move to a different prefix. Media, tool-call history, automatic/request-level markers,
interleaved instructions and other unsupported shapes retain their existing uncached behavior and
disclosures. Cache handling is skipped for body-signed requests, search rounds and repaired payloads.
ZDR requests do not create retained resources.

Vertex returns resource names with a numeric project number, even when the endpoint uses a
project ID. For those endpoints, the host supplies
`GoogleCacheAuthority(vertex_project=VertexCacheProject(endpoint_project="my-project", project_number="123456789"), ...)`
with an independently verified association for that provider account. Missing mapping skips
creation before reserving funds. A conflicting mapping fails closed. Neither the caller nor a
provider response can establish the mapping; numeric endpoints cannot be remapped to another
project. The create URL, model, region and credentials remain exactly those admitted, and only the
verified numeric namespace is accepted for creation results and reuse.

Cache creation happens only after route selection and generation reservation. Rust makes at most
one cache-create HTTP request, using the selected endpoint's credentials, no redirects or retries,
and the remaining request deadline. The request sends a fixed absolute expiration no more than
five minutes away, not a sliding TTL. Response parsing is bounded to 64 KiB and exposes only the
resource name, provider-measured token count, expiration and status to the host callback.

The host owns durable cross-worker claims, customer allowance, credential-generation binding and
resource-cost accounting. Its `claim` must commit the complete create-plus-storage reservation
before granting one creator. Ready resources are isolated by tenant, account generation, endpoint,
project/location, model and exact prefix. A worker-local dictionary is not a durable implementation.
Token storage is priced per million-token-hour, separately from generation/cache-write token legs;
unknown rates are not zero, while an explicitly verified zero create-input rate is representable.

Creation timeouts, cancellations and malformed outcomes retain reserved exposure and never trigger
blind resource recreation. Known but expired resource facts still reach accounting, but the resource
is not reused. Freshness is rechecked after host I/O and again in Rust before use. A failed accounting
acknowledgement prevents generation; an acknowledged unavailable resource can use the original
payload within existing deadlines and generation-attempt limits. No warm-up generation is added.

This engine interface does not install a hosted spending policy, durable resource store, verified
price catalog or customer settings UI. Those must be implemented and tested by the embedder before
activation. Offline and loopback tests do not establish live Google eligibility, realized savings or
provider-side storage billing. No customer is opted in by installing the engine package.

## Verification boundaries

Regression coverage exercises the actual native normalizers, encoders, and served loopback
HTTP sockets. It checks partial tools, quiet disconnects, upstream close before settlement,
terminal precedence, parsed usage preservation, keyed replay, and heartbeat deadline behavior.
These tests do not establish a real provider's cancellation guarantee. A hosted rollout also
requires exact-version provider and ledger verification in its authorized environment.

## Protocol references

- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages)
- [Anthropic thinking controls](https://platform.claude.com/docs/en/build-with-claude/thinking)
- [OpenRouter generation parameters](https://openrouter.ai/docs/api/reference/parameters)
- [OpenRouter streaming and cancellation](https://openrouter.ai/docs/api/reference/streaming)
- [OpenRouter errors and failover](https://openrouter.ai/docs/api/reference/errors-and-debugging)

OpenRouter documents omission of optional upstream token caps, not a universal numeric default.
Its public streaming schema carries argument strings and termination reasons; it does not
specify an internal repair algorithm for every malformed tool response.

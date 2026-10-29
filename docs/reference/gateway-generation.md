# Generation limits and interrupted streams

The gateway preserves caller intent across provider protocols. An omitted optional field is
not permission to select a smaller output budget or a more expensive reasoning mode.

## Output limits

Chat Completions accepts `max_completion_tokens`, `max_tokens`, or `max_output_tokens` through
its protocol decoder. Neither OpenAI nor OpenRouter defines them as exclusive, and many clients
send `max_tokens` beside `max_completion_tokens`, so when several arrive together the smallest
value is the cap (ties keep `max_completion_tokens`, then `max_tokens`) and each other spelling is
disclosed as `max_tokens->ignored(superseded_by_max_completion_tokens)`. Each value must still be
a positive integer. Messages requires `max_tokens`. An explicit limit remains a ceiling.
Admission prefers any rung that accepts the exact value. When no rung on the route does, the
limit is clamped onto the leading rung's accepted range and the change is disclosed: a limit
above the route maximum dispatches at that maximum (`max_tokens->32768(route_maximum)`; Claude
Code sends 64000 or 128000 to every model), and a limit below the route floor (16 on the OpenAI
Responses wire, or a catalog-declared minimum) dispatches at the floor
(`max_output_tokens->16(route_minimum)`). Lowering never bills more than the caller allowed;
raising adds at most the floor's few tokens. A plan account that refuses any ceiling still
narrows out instead.

## Sampling ranges

`temperature` and `top_p` outside the public bounds (`temperature` 0 to 2, `top_p` 0 to 1, never
NaN) are refused by the protocol decoder. Inside them, admission prefers any rung that accepts the
exact value. When every rung that declares the control refuses the value on range, it is clamped
onto the leading rung's range like an output limit and disclosed: Claude's thinking lanes accept
only `temperature` 1.0, so a caller's 0.1 serves as `temperature->1.0(route_minimum)` instead of
a 400. A rung that does not declare the control serves by dropping it
(`temperature->dropped(unsupported_by_provider)`), so its presence leaves the value unclamped.

## Structured output routing

A `response_format` (Chat) or `text.format` (Responses) json_schema request is a hard capability:
admission first narrows the route to the rungs whose catalog declares structured output, in route
order, and only then ranks them by how exactly they keep sampling and thinking preferences.
Otherwise a lead rung that keeps the caller's `temperature` exactly but cannot enforce a schema
would win the ranking and be refused, although a later rung serves the schema (qwen3.8-27b,
2026-10-17). When the lead is passed over the move is disclosed as
`response_format->routed(structured_output_rung)` (`text.format->...` on Responses). A route
with no structured-output rung keeps failing closed on the field. A continuation pinned to the
rung that sealed its reasoning, and a caller-pinned `gateway.routing.route_id`, never move.

## Tool and media routing

Function `tools` and media inputs (images, video, audio, PDFs) are hard capabilities too, and they
narrow before structured output: admission keeps only the rungs whose catalog declares them, then
ranks those by sampling. A rung that does not declare the feature would be refused by the
per-rung capability check anyway, so narrowing never drops a rung that could serve; it stops a
tool-less rung that keeps `temperature` and `top_p` exactly from winning the ranking. GitHub
Copilot sends tools with `temperature` 0.1 and `top_p` 1; on gpt-6-astra only the tool-less Novita
rung takes those controls, so the request was refused as unsupported `tools` although the OpenAI
rung serves tools (2026-10-20). That rung now serves and drops the sampling it does not take,
disclosed. A passed-over lead is disclosed as `tools->routed(tools_rung)`, or
`messages->routed(image_input_rung)` (`input->...` on Responses) for media. Because structured
output narrows among the rungs tools kept, a request no rung can serve whole is refused on the
feature the tool rungs lack (`response_format`), not on `tools`. A route with no rung for the
feature keeps failing closed on it.

## Output limits and context windows

A limit that no rung's declared context window can reserve beside the prompt (`max_tokens`
10,000,000 on a 131,072-token route that declares no output maximum) is lowered before the
window check to what the largest window leaves, disclosed as
`max_tokens-><n>(context_window)`. The prompt is sized from above for this: one token per text
byte plus a 256-token template reserve, or three bytes per token when that leaves no room. A
rung that declares no window, or that already holds the request, keeps the value. Only the
output limit changes: a prompt that alone exceeds every window is still refused with
`context_length_exceeded`.

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
message replayed with `status` or `phase` but without its item ID (LiteLLM's
Responses bridge echoes history this way) decodes as plain assistant history:
the markers cannot anchor a replay without the ID, so they are dropped and
disclosed as `input.status->dropped(no_item_id)` / `input.phase->dropped(no_item_id)`.
Messages that carry their ID keep status and phase for exact replay.

Item IDs the OpenAI Responses API could not have issued (a client's `msg_<uuid>`,
another gateway's `item_` prefix, a non-`fc_` function-call ID) are never sent to
an OpenAI Responses rung, which refuses them. The item replays without its ID at
its original position; `store=false` means the provider resolves nothing by ID.

Codex records output items in the order they complete, so a replayed turn can
carry an assistant message between its `function_call` items and their
`function_call_output` items. Chat Completions requires the tool results to follow
the message that called them (DeepSeek answers otherwise with HTTP 400), so a Chat
rung folds that text into the tool-call message's content, or moves it ahead of the
calls when it cannot merge losslessly, and the route discloses
`messages.assistant.content->tool_call_turn`. The Responses wire keeps the items
verbatim. The gateway's own Responses stream completes items (`output_item.done`)
in the order `response.completed.output` lists them, so a recorded history matches
the order the model produced.

## Tolerated client fields

- Responses `stream_options` is accepted. The gateway never emits obfuscation
  padding, so `include_obfuscation: false` is honored exactly; `true` is disclosed
  as `stream_options.include_obfuscation->ignored(not_emitted)`, and the object on
  a non-streaming call as `stream_options->ignored(stream_false)`.
- Responses `text.format.type: "json_object"` selects the same schema-free JSON mode
  as Chat `response_format: {"type": "json_object"}`.
- A key the gateway does not define on a conversation object is dropped before dispatch
  instead of refused: on Chat messages, their assistant `tool_calls` entries, their
  content parts and each part's media object (`image_url`, `video_url`, `input_audio`,
  `file`; the Vercel AI SDK sends `image_url.providerOptions`), on Responses `message`
  input items and their content parts, and on Messages turns (Claude Code 2.1.282 and
  later sends `output_config` inside a turn, disclosed as
  `messages.output_config->dropped(unknown_message_field)`; the top-level `output_config`
  is unaffected). Chat apps stamp their own bookkeeping on the history they replay
  (WorkBuddy's `conversationRequestId`, `isInternalToolCall`, `model_id`), so a refusal
  failed every later turn. Each dropped key is named once per request, as
  `messages.<field>`, `messages.tool_calls.<field>`, `messages.content.<field>` or
  `messages.content.image_url.<field>` (`input.` on Responses)
  `->dropped(unknown_message_field)`. Keys traced to one client keep
  `client_annotation` as the reason: Copilot Chat's `cot_id` and `cot_summary`, the
  `timestamp` and `startsNewUserRequest` other agents stamp, WorkBuddy CLI's `agent`, and
  an image's `image_url.dimensions` size hint. Field names are reduced to `[A-Za-z0-9_]`, at most 64
  characters, and at most 16 keys are named; the rest are counted in one
  `messages.*->dropped(<n>_more_unknown_message_fields)` entry. Defined fields keep
  their typed refusal for a bad value, unknown keys inside a tool call's `function`
  object, inside a Messages content block, or on other Responses item types stay
  refused, and an unknown top-level parameter is still `unsupported_parameter`.
- OpenAI's deprecated (not removed) function-calling shapes on Chat
  ([API reference](https://platform.openai.com/docs/api-reference/chat/create)) are rewritten
  to their `tools` spelling before dispatch: an assistant `function_call: {"name",
  "arguments"}` becomes one `tool_calls` entry with the id `call_legacy_function_<message
  index>`, disclosed as `messages.function_call->translated(tool_calls)`; a `{"role":
  "function", "name", "content"}` result becomes a `tool` message answering the nearest
  earlier unanswered call of that name, keeping its `name`, disclosed as
  `messages.role->translated(function_as_tool)` (a documented `content: null` result is sent
  as `""`, `messages.content->translated(null_function_result_as_empty)`); the `functions`
  parameter becomes `tools` (`functions->translated(tools)`) and `function_call: "none" |
  "auto" | {"name"}` becomes `tool_choice` (`function_call->translated(tool_choice)`). What
  cannot be paired exactly is a 400 `invalid_parameter` naming the field: a result whose name
  matches no waiting call (`messages.N.name`), a call no result answers or one beside
  `tool_calls` (`messages.N.function_call`), and `functions` / `function_call` beside `tools` /
  `tool_choice`.
- OpenAI's optional participant `name` on a system, developer, user or assistant message is
  dropped, disclosed as `messages.name->dropped(participant_name)`. On tool messages `name`
  still attributes the result.
- A Chat system, developer, user or tool message whose `content` is null or omitted (Go clients
  drop an empty tool result under `omitempty`) is refused with `400 invalid_parameter` naming
  `messages.<n>.content`; an empty string is content and is accepted.
- `chat_template_kwargs.preserve_thinking` is accepted and disclosed as
  `chat_template_kwargs.preserve_thinking->ignored(not_forwarded)`: no wire receives
  `chat_template_kwargs` verbatim.
- A Responses reasoning input item with neither `id` nor `encrypted_content` cannot be resolved
  by any provider, and its summary never reaches provider history, so it is removed,
  `input.reasoning->dropped(no_id_or_encrypted_content)`.
- Codex Desktop's `internal_chat_message_metadata_passthrough` on Responses input items is
  removed, `input.internal_chat_message_metadata_passthrough->dropped(client_annotation)`, and
  its `stream_options.reasoning_summary_delivery` is disclosed as
  `stream_options.reasoning_summary_delivery->ignored(not_supported)`.
- Inline input images are accepted up to 20 MiB decoded (OpenAI's per-image limit; the base64
  text may be up to 27,962,028 characters). Narrower provider limits (Anthropic, Bedrock)
  surface as provider errors on those rungs.
- On the Messages surface a `document` block with Anthropic's `text` source
  (`media_type: "text/plain"`) joins the turn as the caller's text, title first. A content
  block of a type the surface does not define is refused naming that type.
- A control-store refusal that names no caller field (for example a model-chain release
  crossing) answers a retryable 503 `gateway_authority_unavailable` with `Retry-After: 5`
  (`overloaded_error` on the Messages surface), with the store's reason in the worker log,
  instead of an unnamed 400.
- `service_tier: "priority"` (Codex Fast mode) on a route with no priority-capable
  rung dispatches at the standard tier, disclosed as
  `service_tier->default(priority_unavailable)`. Standard never costs more. `flex`
  asks for a cheaper tier, so it stays a typed refusal on such a route.

## Reasoning controls

When no rung on a route can honor a reasoning control verbatim, admission applies the
nearest servable reading and discloses it:

- `none` or `minimal` below a route's whole effort ladder (for example `low, high, max`)
  takes the shallowest tier the route serves, `reasoning_effort->low(lowest_supported)`.
  Such a route cannot turn reasoning off; other efforts still only snap down.
- A level-less enable-thinking toggle (`enable_thinking`, `chat_template_kwargs`) on a
  route whose rungs cannot reason at any depth is dropped,
  `enable_thinking->dropped(no_reasoning_rung)`. Numerical thinking budgets are never
  dropped this way.
- `include: ["reasoning.encrypted_content"]` on a route with no native Responses rung and no
  gateway reasoning carrier is dropped, `include.reasoning.encrypted_content->dropped(no_reasoning_carrier)`:
  the answer is what an omitted selector produces. Replayed native encrypted reasoning that no
  rung can read is removed from history, `input.reasoning.encrypted_content->dropped(foreign_route)`.
  A route with a native Responses rung keeps the payload and the provider judges it.

Valid explicit thinking-off settings remain off. Model families distinguish support for
budgeted thinking from support for disabling thinking. Claude Opus 5.5, Fable and Mythos
always reason: no effort level turns thinking off. On these models an explicit
`thinking.type: disabled` dispatches as the provider's own substitute for "off", adaptive
thinking at effort `low` with the reasoning withheld (`display: "omitted"`), and the response
discloses `thinking.type->adaptive` (plus `output_config.effort->low` when the request named
another effort). Clients that do not recognize the model send the off switch anyway: Claude
Code's session-title call does on the gateway's dotted aliases, and a refusal failed every one.
Any other unsupported off setting stays a typed pre-dispatch refusal (Opus 5 at `xhigh` or
`max`, Sonnet 5.5 at any effort), never a silent switch to the model's default thinking.
Opus 5.5 also cannot force a tool selection: the existing capability policy uses `auto`
only after all eligible routes decline the forced choice and discloses `tool_choice->auto`.
This does not guarantee a tool call. See the provider's
[Opus 5.5 contract](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide).

Some origins turn thinking off with their own switch rather than `reasoning_effort: none`.
GLM-5 and GLM-5.1 on Z.ai and on Huawei Cloud MaaS take `thinking: {type: disabled}` for
`none` and `{type: enabled}` for any other level. Huawei MaaS DeepSeek-V4 models take
`thinking: {type: disabled}` for `none` and keep `reasoning_effort` for `high` and `max`.
Vertex MaaS GLM takes `chat_template_kwargs: {enable_thinking: false}`. On OpenRouter, a
request with reasoning off also lists the upstreams seen to keep thinking anyway in
`provider.ignore` (AtlasCloud today), so OpenRouter serves it elsewhere or refuses.

A Responses `reasoning.summary` (or its deprecated alias `reasoning.generate_summary`) is
best effort, as on OpenAI's own API. Admission prefers a rung whose wire carries summary parts over every other preference
(native Responses, or Anthropic thinking projected onto them). A route with no such rung (for
example gpt-5-nano's Azure Chat, Novita and OpenRouter lanes) serves the turn without summaries
and discloses `reasoning.summary->dropped(unsupported_by_provider)` under the caller's spelling.

An OpenAI `allowed_tools` tool choice (Chat `{"type": "allowed_tools", "allowed_tools": {...}}`
or Responses `{"type": "allowed_tools", "mode": ..., "tools": [...]}`) restricts the model to
the listed functions. A route whose every rung is native Responses forwards the selector verbatim,
keeping the full tool list for prompt caching. Any other route declares only the allowed
functions under the plain mode (`auto` or `required`, which each wire maps to its own `auto`,
`any` or `ANY`) and discloses `tool_choice->translated(allowed_tools_as_restricted_tools)`. A
`required` set on a model that cannot force tools relaxes to `auto` under the forced-choice
policy above, keeping the set. Chat lists only functions; Responses may also list the SDK's
non-function entries (`mcp`, `custom`, `image_generation`), forwarded verbatim on native routes
and matched to the declarations they name elsewhere. When nothing the set allows survives on a
wire, the choice is cleared with `tool_choice->cleared(no_serviceable_tool)`. Gateway-run tool
search narrows its corpus to the allowed set before the first round.

An explicit effort cannot be coerced upward. If a route needs a supported spelling, only an
admissible lower tier may be selected, with disclosure; a reasoning-capable route with no
admissible setting is refused. A genuinely non-reasoning route retains its existing disclosed
omission policy. Omitted thinking and effort remain omitted unless the provider protocol requires
an authored setting. A model name appearing inside a system message is not an API setting.

A caller that explicitly requests budgeted thinking without a budget receives a derived
budget that must fit the effective per-provider output ceiling. An explicit budget is not
rewritten to fit. An impossible combination is refused before provider dispatch. Numeric
Messages budgets are forwarded verbatim to budget-capable wires. On a route with no
budget-capable rung, the budget snaps to the nearest tier the route's effort ladder serves
(under 2k tokens `low`, under 8k `medium`, under 24k `high`, above that `xhigh`; `0` asks for
off and `-1` for the lane's default depth), disclosed as
`thinking.budget_tokens->reasoning_effort:<tier>(route_effort_ladder)`. Only a route with no
reasoning tier refuses it, and a budget beside an explicit effort stays refused. Haiku 4.5
rejects adaptive thinking and `output_config.effort` by name, so `thinking: {type: "adaptive"}`
there becomes a budget derived from `max_tokens` (`thinking.type->enabled`), and a recognized
effort beside it drops as `output_config.effort->dropped(superseded_by_thinking_budget)`.

The Chat field `thinking_budget` preserves the caller's numeric control on qualified
Anthropic, Gemini 2.5, and native Qwen Cloud routes. Use it as an OpenAI SDK `extra_body`
field (or a top-level field in HTTP JSON). It is not combined with `reasoning_effort`.
The gateway translates the field name for the selected provider and discloses that
translation. On a route with no qualified rung it snaps onto the effort ladder like a
Messages budget, disclosed as `thinking_budget->reasoning_effort:<tier>(route_effort_ladder)`.

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

OpenRouter's deprecated `include_reasoning` boolean is accepted on Chat as the alias its
[reasoning tokens guide](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
documents: `false` becomes `reasoning.exclude: true` (disclosed as
`include_reasoning->translated(reasoning.exclude)`, then under `exclude`'s own disclosed
omission policy, `reasoning.exclude->dropped(not_carried)`), while `true` (the default) and `null`
are consumed silently. A value contradicting an explicit `reasoning.exclude` is a 400 naming both.

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
authorization separately from settled spend. On a capped request, an OpenAI-compatible `stop`
with no usage report is read as a budget truncation: the caller sees `length`, and the dispatched
incomplete terminal carries the internal `usage_unreported_truncation` signal with the same
`streamed_output`. An opened one is completed by the same estimate, with output metered at no less
than the output budget the attempt dispatched with, because the provider generated and billed the
whole budget, unstreamed reasoning included. Meta's muse lanes end a stream that hit `max_tokens`
with `finish_reason: stop` and no usage frame (2026-10-17), and those turns had settled at $0. Any
other observed terminal with missing usage retains the host's unknown-terminal policy.
Stopping generation is not evidence that its unreported usage was free, and an estimate is never
reported as observed.

Meter parsing follows the provider's wire contract. Missing primary counts on partial
OpenAI, Anthropic, or Bedrock reports remain unknown. Gemini's present `usageMetadata` uses
implicit-presence protobuf scalars, so omitted zero-valued counts retain Google's zero-default
meaning; thinking tokens still contribute to billed output. An absent usage object is unknown.
Cumulative reports from one generation are merged, not added. Costs from separate physical
generations cannot acquire a known total by adding a known count to an unknown one.

Chat and Responses preserve the observed `cache_write_1h_tokens` detail within total cache
writes. An omitted split stays unknown; a measured zero stays zero. Where the standard response
schema requires an integer, `usage.unreported_token_details` names any compatibility zero that
stands for an unreported meter. Its only allowed names are `cached_tokens`, `cache_write_tokens`,
`cache_write_1h_tokens`, and `reasoning_tokens`, without duplicates. A named detail must be absent,
null, or the integer zero; contradictory positive or malformed values are rejected. Native
gateway hops and SDK provider adapters restore those details to unknown before valuation.
Every relay hop must preserve this evidence. A downstream parser cannot recover observability
that an older relay already discarded.

OpenAI's 12-hour prompt cache lifetime rides Responses `prompt_cache_options.ttl: "12h"`
([OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching): "In the
Responses API, set `prompt_cache_options.ttl` to choose the minimum cache lifetime"). `"30m"` is
the provider default and is accepted as already satisfied; any other value, and `"12h"` on Chat
(OpenAI's Chat answers 400 for it; `"30m"` is Chat's only documented lifetime), is refused. Admission
narrows the route to rungs declaring `supports_prompt_cache_ttl_12h` whose wire carries the field
(OpenAI's own Responses API) and refuses the request with `unsupported_capability` on
`prompt_cache_options.ttl` when none does, so no rung that would write at the default lifetime or
refuse the field (Azure, OpenRouter, a ChatGPT plan rung, a pre-GPT-5.6 model) is ever asked. An organization
with prompt caching off has the lifetime removed and disclosed. The provider's
`input_tokens_details.cache_write_tokens_details.ttl_12h_tokens` becomes the 12-hour write
subset, disjoint from any one-hour subset, and is relayed as `cache_write_tokens_details` with
`ttl_30m_tokens` and `ttl_12h_tokens`. Those writes price at the card's
`cache_creation_12h_input_nano_usd_per_million_tokens`, or at 2.5 times its input rate (OpenAI's
published total write rate) when the card authors none, never at the 30-minute write rate. A
12-hour request whose provider reports writes without the split bills every write at 12 hours.

`prompt_cache_options.mode: "explicit"` and the content-block `prompt_cache_breakpoint`
(`{"mode": "explicit", "ttl"?: ...}`) are accepted on both OpenAI surfaces
(`exp/runtime/openai_protocol/prompt_cache_controls.py`). `mode: "implicit"` is the provider
default and is never forwarded; any other mode, a breakpoint that is not that exact shape, and a
breakpoint `ttl` other than the request's own lifetime (re-emitted breakpoints inherit it) are
named 400s, as is `prewarm`. A text-block breakpoint rides the canonical ephemeral
`cache_control` carrier at the same block, so per dialect:

- OpenAI, Azure OpenAI (its `openai_deployments` surface) and OpenRouter, on Chat and Responses,
  receive `prompt_cache_options` and every breakpoint in their own spelling
  (`GatewayWireProfile.forwards_prompt_cache_options`; a marked instruction rides an input item,
  since Responses `instructions` cannot hold one). Explicit-mode cache writes are reported and
  billed through the same `cache_write_tokens` legs as implicit ones.
- Anthropic and Bedrock cache at the same blocks as ephemeral `cache_control` / `cachePoint` at
  their 5-minute default, never the dearer 1-hour lifetime, disclosed as
  `prompt_cache_breakpoint->translated(cache_control_5m)`; they cache only at markers, so
  explicit mode already holds there.
- Every other wire (Gemini, Novita, vLLM and other OpenAI-compatible servers, Azure's Foundry
  model-inference surface) never receives the fields. Like any cache marker they are kept while
  some rung honors them, and only a route where none does discloses
  `prompt_cache_options.mode->not_forwarded(provider_decides_caching)` and
  `messages.content.prompt_cache_breakpoint->not_forwarded(...)`. A breakpoint on an image,
  audio or file part, or inside a Responses tool output, is dropped with a
  `->dropped(unsupported_media)` / `->dropped(unsupported_tool_output)` disclosure.

Anthropic Messages and Bedrock Converse bill thinking inside `output_tokens` and publish no
separate count, so their reasoning meter stays unknown for settlement. Chat and Responses
callers still receive an estimate, the way OpenRouter reports one: when the provider published
no count and the turn carried thinking in any form, `completion_tokens_details.reasoning_tokens`
(Chat) or `output_tokens_details.reasoning_tokens` (Responses) is `output_tokens` less
`ceil(visible_bytes / 4)` for the answer text, refusal text and client tool arguments, floored
at zero. The usage object then names it in `usage.estimated_token_details`
(`["reasoning_tokens"]`, its only allowed entry) instead of `unreported_token_details`. The
estimate is display only: it never changes the settled usage, and native hops and SDK provider
adapters read a marked count as unknown.

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
verified numeric namespace is accepted for creation results and reuse. Native Vertex profiles
retain the receipt of the atomically resolved service-account credential so a host can bind the
resource to that account generation. Refreshing an OAuth bearer does not replace the source receipt;
resolving a rotated service account does. This receipt does not enable static-auth cache-affinity
recovery for Vertex, including when the host declares an operational region.

Cache creation happens only after route selection and generation reservation. Rust makes at most
one cache-create HTTP request, using the selected endpoint's credentials, no redirects or retries,
and the remaining request deadline. The request sends a fixed absolute expiration no more than
five minutes away, not a sliding TTL. Response parsing is bounded to 64 KiB and exposes only the
resource name, provider-measured token count, expiration, optional creation time and status to the
host callback. `CacheResult.create_time` carries Google's `createTime` as an absolute Unix timestamp,
not a TTL or locally inferred start time. A known value is positive, finite and no later than the
reported expiration or its observation; missing, malformed or future creation facts remain `None`.
A valid resource can remain ready with `create_time=None`: readiness establishes usability, not
billing completeness. A host whose published customer schedule requires creation time must retain
its full hold until it has sufficient provider facts and may refuse accounting acknowledgement.
The provider-reported resource interval is evidence, not an exact invoice formula or proof of
provider billing granularity, garbage-collection timing or realized storage charges.

The host owns durable cross-worker claims, customer allowance, credential-generation binding and
resource-cost accounting. Its `claim` must commit the complete create-plus-storage reservation
before granting one creator. Ready resources are isolated by tenant, account generation, endpoint,
project/location, model and exact prefix. A worker-local dictionary is not a durable implementation.
Token storage is priced per million-token-hour, separately from generation/cache-write token legs;
unknown rates are not zero, while an explicitly verified zero create-input rate is representable.
The 300-second quote applies the host-authored customer schedule and bounds that authorized
reservation; it does not establish an upper bound on every possible provider invoice adjustment.

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

## Provider routing preferences

Every surface accepts OpenRouter's documented `provider` object
([provider routing](https://openrouter.ai/docs/features/provider-routing)). The gateway resolves
its routing keys against the model's own rungs at admission and never forwards them to an
aggregator upstream:

- `only` keeps rungs of the listed providers; `ignore` drops them.
- `order` tries the listed providers first, in that order; the rest remain fallbacks.
- `allow_fallbacks: false` with `order` serves only the listed providers; without `order` it
  serves only the first eligible rung. A pinned rung that fails, or is unavailable, is the
  request's error, never a failover.

A rung is named by its catalog provider id (`bedrock`, `azure_openai`, `openai`, `anthropic`,
`openrouter`, ...), or by the label a host assigns through its components' optional
`provider_label` callable (a hosted platform maps a masked house lane to the name it shows, such
as `experiential_cloud`, so callers pin the name they see in `x-gateway-provider`). OpenRouter
slugs are aliases: `amazon-bedrock`, `azure`, `google-vertex`, `google-ai-studio`. A preference
no rung satisfies (an unknown name, or `only`/`ignore` excluding every rung) is a 400 on
`provider.<key>` naming the providers serving the model. `zdr` keeps its zero-data-retention
handling; `zdr` and `data_collection` are the only caller keys an OpenRouter rung receives. Any
other key (`sort`, `quantizations`, `max_price`, `require_parameters`, ...) is disclosed as
`provider.<key>->ignored(unsupported)`. `gateway.routing` (`route_id`, `allow_fallbacks`)
applies on top of the narrowed route.

## Retained provider output

Provider output the gateway holds while relaying (tool arguments and reasoning summaries in the
normalizer, Messages blocks held for ordering, a non-streaming aggregation, a Responses
continuation) is bounded per stream at 64 MiB. An embedder may also bound it across all streams
of one worker with `maximum_retained_output_bytes` (`serve_native_gateway`). The worker-wide
budget is off by default, which is the behavior before it existed; 1 GiB, a quarter of a 4 GiB
worker, is the suggested value once enabled. With it on, a stream whose next retained bytes would
cross the budget fails with a retryable 503 `gateway_unavailable`, and a Responses continuation
that would cross it is not retained while the stream serves on. Bytes relayed straight through to
the caller never count, and every reservation returns when its stream ends.

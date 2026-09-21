# Generation limits and interrupted streams

The gateway preserves caller intent across provider protocols. An omitted optional field is
not permission to select a smaller output budget or a more expensive reasoning mode.

## Output limits

Chat Completions accepts `max_completion_tokens` and `max_tokens` through its existing
protocol decoder. Messages requires `max_tokens`. An explicit limit remains a ceiling:
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

## Reasoning controls

Valid explicit thinking-off settings remain off. An unsupported off setting is a typed
pre-dispatch refusal, not a request to use the model's default thinking behavior. Model
families distinguish support for budgeted thinking from support for disabling thinking.

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

Numeric reasoning budgets on Chat (`reasoning.max_tokens` or `thinking.budget_tokens`) are
refused rather than approximated by an effort tier. Use Messages with a budget-capable model,
or deliberately remove the numeric budget and select an effort. This is a documented
compatibility limitation, not a claim of full OpenRouter reasoning-parameter parity.

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
even when partial counts are known. Hosted monetary accounting retains unresolved authorization
separately from settled spend for later resolution. The local gateway's monthly allocation uses
its existing conservative policy instead: the full reserved bound consumes budget, the provider
cost stays unknown, and observed partial counts remain available. That local allocation is shown
under settled budget and is not automatically reconciled; it is not a provider bill or a hosted
customer-account debit. An observed terminal with missing usage retains the host's unknown-terminal
policy; this marker does not broaden that policy. Stopping generation is not evidence that its
unreported usage was free.

Meter parsing follows the provider's wire contract. Missing primary counts on partial
OpenAI, Anthropic, or Bedrock reports remain unknown. Gemini's present `usageMetadata` uses
implicit-presence protobuf scalars, so omitted zero-valued counts retain Google's zero-default
meaning; thinking tokens still contribute to billed output. An absent usage object is unknown.
Cumulative reports from one generation are merged, not added. Costs from separate physical
generations cannot acquire a known total by adding a known count to an unknown one.

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

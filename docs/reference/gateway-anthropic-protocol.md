# Anthropic Messages protocol

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
is refused before dispatch. A numeric thinking budget no rung carries takes the disclosed
`route_effort_ladder` snap. A bare `{type: enabled}` without a numeric budget can use a disclosed derived
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

## Stream usage

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
carries the provider's final meters plus the gateway's cost extensions; the official Anthropic
SDK accumulators (Python and TypeScript) copy every usage field present on `message_delta`, so
their final message shows the true counts. The estimate is display-only: the encoder keeps it
apart from the usage it settles from, so it never reaches `message_delta` or the ledger, which
bill the provider's report.

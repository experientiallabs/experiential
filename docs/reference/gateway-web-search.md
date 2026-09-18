# Gateway-executed web search

Gateway-executed web search covers every route that has no provider-native search. A caller
asks for it in any of the public spellings (Chat `web_search_options`, the OpenRouter
`plugins: [{"id": "web"}]` object or `:online` model suffix, a Responses `web_search` /
`web_search_preview` tool, an Anthropic `web_search_*` server tool); each decodes to one
canonical `GatewayRequest.web_search` (`exp/runtime/gateway/web_search/contracts.py`), which
joins replay identity like every other answer-changing field. After the route is admitted
(`exp/runtime/gateway/web_search/plan.py`): a route whose every rung serves the spelling natively
(all-Responses for the Responses tool, all-Anthropic for the server tool) keeps the provider's own
search untouched; otherwise the control plane runs ONE search against the configured backend
(`exp/runtime/gateway/web_search/backend.py`; Exa when `EXA_API_KEY` is set, or a host-supplied
backend on `NativeControlPlane(web_search=...)`), bounded by eight seconds and the request
deadline, for the latest user turn's text. The ranked results are injected as one instruction
`system` turn placed after the leading instructions (never `developer`: a compatible rung may lack
that capability),
so token reservations already count them and every wire dialect carries them; the
provider-native search carriers are stripped so a mixed route answers from the same evidence.
Admission then tells the data plane `{"web_search": {"query", "requests", "results": [{"url",
"title"}]}}`, and the encoders cite every result URL that appears in the final text:
`url_citation` annotations on Chat (`choices[].message.annotations`, one trailing
`delta.annotations` chunk when streaming) and Responses (`output_text.annotations` plus
`response.output_text.annotation.added` events), and synthesized `server_tool_use` /
`web_search_tool_result` blocks ahead of the answer on Messages. The count reaches settlement as
`web_search_requests` (`usage.server_tool_use_details.web_search_requests` on the OpenAI
surfaces, `usage.server_tool_use.web_search_requests` on Messages) so a host can bill the search.
Without a backend, or when the vendor fails or times out, the turn still serves and the drop is
disclosed (`web_search->dropped(search_unavailable|search_failed|no_query)`); a search never
fails a request.

## Request spellings

| Surface | Spelling | Native pass-through when |
|---|---|---|
| Chat Completions | `web_search_options: {search_context_size, user_location}` | never (the gateway always runs it) |
| Chat Completions | `plugins: [{"id": "web", "max_results", "search_prompt", "include_domains", "exclude_domains"}]` | never |
| Chat + Responses | model name suffix `:online` (stripped before alias authorization) | never |
| Responses | `tools: [{"type": "web_search" | "web_search_preview" | dated variants, "search_context_size", "filters": {"allowed_domains"}}]` | every rung is `openai_responses` |
| Messages | `tools: [{"type": "web_search_20250305" (or newer), "name", "max_uses", "allowed_domains" \| "blocked_domains", "user_location"}]` | every rung is `anthropic_messages` |

`search_context_size` maps to 3 / 5 / 8 results (`low` / `medium` / `high`); the plugin's
`max_results` wins when present; the ceiling is 10. Allowed and blocked domains are mutually
exclusive (the provider's rule) and are forwarded to the vendor as include/exclude filters.

## Response shapes

* Chat: `choices[0].message.annotations[]` of `{"type": "url_citation", "url_citation": {"url",
  "title", "start_index", "end_index"}}` (character offsets into the final text, one entry per
  occurrence of a result URL); when streaming, one chunk carrying `delta.annotations` precedes the
  finish chunk. `usage.server_tool_use_details.web_search_requests` counts the search.
* Responses: the message's `output_text` part carries flat `{"type": "url_citation", "url",
  "title", "start_index", "end_index"}` annotations, each announced by a
  `response.output_text.annotation.added` event before `response.output_text.done`;
  `usage.server_tool_use_details.web_search_requests` counts the search.
* Messages: a `server_tool_use` block (`name: "web_search"`, `input: {"query"}`) and a
  `web_search_tool_result` block (`web_search_result` entries with `url`, `title`, an empty
  `encrypted_content`, `page_age: null`) precede the answer; `usage.server_tool_use.web_search_requests`
  counts the search.

## Configuration and disclosure

`EXA_API_KEY` binds the Exa backend (`EXA_SEARCH_URL` may point at a proxy or a loopback fixture;
only https or loopback origins are accepted). A host may pass any `WebSearchBackend` to
`NativeControlPlane(web_search=...)`. Without a backend the ask is dropped with
`web_search->dropped(search_unavailable)`; a vendor failure or timeout (8 s, bounded by the request
deadline, enforced by the gateway even for a backend that ignores it) yields
`web_search->dropped(search_failed)`; a request with no user text yields
`web_search->dropped(no_query)`. A `tool_choice` that named the replaced search tool (or demanded
"any tool" when the search was the only one) is cleared with
`tool_choice->cleared(no_serviceable_tool)`. In every case the turn still serves and no search is billed.

## Settlement

The data plane reports `web_search_requests` on the finalizing settlement only, so a failed rung
followed by the winner never bills the search twice; the hosted ledger receives it through the
feature-probed `finish_attempt(web_search_requests=...)` keyword and prices it. The engine's local
SQLite ledger accepts the count and does not yet price it.

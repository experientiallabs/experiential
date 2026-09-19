# Gateway-executed tool search

A caller with a large toolset marks most tools `defer_loading: true` and declares a tool-search
tool in the provider's own spelling. On a route every rung of which serves that spelling natively
(all-Anthropic for `tool_search_tool_bm25` / `tool_search_tool_regex`, all-OpenAI-Responses for
`{"type": "tool_search"}`) the declaration and the deferred markers forward verbatim and the
provider searches. On every other route the GATEWAY runs the search:

1. Admission (`exp/runtime/gateway/tool_search/plan.py`) partitions the caller's tools: those
   without `defer_loading` load immediately; the rest form the searchable corpus. The model
   receives the loaded tools plus one gateway-owned function tool named `tool_search`
   (`gateway_tool_search` if the caller already used that name) whose parameters are `query`
   (natural language, BM25), `pattern` (regular expression) and `limit` (1-10). Which of `query`
   and `pattern` appear follows the caller's declaration (`bm25`, `regex`, or both for the
   OpenAI and OpenRouter shapes). The provider-native declarations are stripped; a `tool_choice`
   that named one is cleared with `tool_choice->cleared(no_serviceable_tool)`; a declaration with
   no deferred tools has nothing to search and is disclosed as
   `tool_search->dropped(no_deferred_tools)`.
2. The data plane withholds every `tool_search` call the model makes (the caller never sees it).
   A dial whose only output was such calls does not commit the rung; the waterfall asks the
   control plane for the next dispatch (`tool_search_round`), which runs the search over the
   deferred corpus (`exp/runtime/gateway/tool_search/search.py`: a small BM25 with a
   lowercase alphanumeric tokenizer that splits camelCase and snake_case, or `re.search` with a
   200-character pattern bound), appends the assistant call and a tool result naming the matched
   tools, moves them into the loaded set, rebuilds that rung's wire entry from the retained
   admission material, and answers with the new wire. The same rung is re-dialed as a fresh
   attempt at the same depth (`dispatch_reason: tool_search_round`), up to `max_rounds` (3) per
   request and never beyond the request's total attempt ceiling. On the last permitted round the
   search tool is withdrawn so the model must answer. A search call arriving after other output
   already committed the rung is dropped and disclosed as `tool_search->dropped(after_output)`.
3. The final answer renders the round trip ahead of the text: on Messages a `server_tool_use`
   block (`name` = the caller's declaration type, `input` = `{"query"}` or `{"pattern"}`) and a
   `tool_search_tool_result` block whose `content.tool_references` name the matched tools; on
   Responses `tool_search_call` / `tool_search_output` hosted items (the output lists the matched
   declarations); on Chat nothing. Usage carries the round count
   (`server_tool_use_details.tool_search_requests` on the OpenAI surfaces,
   `server_tool_use.tool_search_requests` on Messages) and settlement reports
   `tool_search_requests` on the finalizing settlement so a host can meter it.

## Request spellings

| Surface | Declaration | Deferred marker |
|---|---|---|
| Messages | `tools: [{"type": "tool_search_tool_bm25" \| "tool_search_tool_regex" (or the `_20251119` variants), "name": ...}]` | `defer_loading: true` on custom tools |
| Responses | `tools: [{"type": "tool_search"}]` | `defer_loading: true` on function tools |
| Chat | `tools: [{"type": "openrouter:tool_search"}]` | `defer_loading: true` on the tool object (beside `function`) |

## Next-turn history

Callers replay the gateway's own output. Messages accepts `tool_search_tool_result` and
`tool_reference` blocks in assistant history (carried verbatim to Anthropic rungs); Responses
accepts `tool_search_call` / `tool_search_output` items, which a foreign wire receives as the
function call and tool result they were (`codex_tools.convert_native_history`).

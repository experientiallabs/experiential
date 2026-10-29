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

## Client-side tool search (`tool_reference` in a tool result)

A Messages caller can run its own search instead of declaring a server tool (Claude Code's
ToolSearch does this): it marks most tools `defer_loading: true`, declares an ordinary custom
search tool, and answers that tool's call with a `tool_result` whose content holds one
`{"type": "tool_reference", "tool_name": "<name>"}` block per deferred tool it loads. On
Anthropic's API a deferred tool stays out of the model's context until a reference loads it,
and the API expands each reference into the named tool's full definition.

The decoder admits `tool_reference` sub-blocks beside text, image and document sub-blocks (an
optional `cache_control` on one re-emits on the enclosing result, the same prompt boundary) and
carries them on the canonical tool message's `tool_references`. Any other sub-block type is still
a named 400. Each wire then maps them as follows (`exp/runtime/models/providers/tool_references.py`):

| Dialect | `tool_reference` | Deferred tools |
|---|---|---|
| `anthropic_messages` (Anthropic, and every rung that speaks its wire) | re-emitted as reference blocks after the result's own content (a references-only result is byte-identical) | every tool forwards with `defer_loading` intact; the provider expands references |
| Chat, Responses, Bedrock Converse, Gemini | one text line per reference in the tool result (`Tool loaded: <name>. ...`), disclosed as `messages.content.tool_result.tool_reference->text` | every tool dispatches immediately loaded, with its full definition and no marker, disclosed as `tools.defer_loading` |

The same rule holds for any request with deferred tools and no tool-search declaration, on every
surface: a non-Anthropic rung never receives `defer_loading`. OpenAI's own marker needs its hosted
`tool_search` in the same request (it refuses `Invalid Value: 'tools.defer_loading'. Deferred
tools require tools.tool_search.`), and a caller searching client-side declares none. Loading every
tool never fails the request, leaves every declared tool callable, and keeps the tool prefix stable
across turns so provider prompt caching holds; the cost is the deferred definitions' input tokens.
A request that does declare a tool-search tool takes the planner above instead: an all-native
route keeps the provider's search and its markers, and any other route runs the gateway search,
which partitions the tools and clears the markers itself. Only the reference text applies to it.

## Mid-conversation tool changes (`tool_addition` / `tool_removal`)

Anthropic lets a `role: "system"` turn inside `messages` change which tools the model is offered
from that point on, without editing `tools`
([Mid-conversation system messages and tool changes](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages)).
The blocks sit in the system turn's `content`, beside `text` blocks:

| Block | Shape | Meaning |
|---|---|---|
| `tool_addition` by reference | `{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "<name>"}}` | offer a tool declared in `tools` (usually `defer_loading: true`), or re-offer a removed one |
| `tool_addition` by value | `{"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {<custom tool>}}}` | define a tool (beta `inline-tools-2026-09-15`); a later definition under the same name replaces it |
| `tool_removal` | `{"type": "tool_removal", "tool": {"type": "tool_reference", "name": "<name>"}}` | withdraw a tool from that point on |

Claude Code sends them when MCP tools arrive late or deferred tools load: the system turn's
reminder text, then one removal per withdrawn tool, then one addition per added tool, the cache
marker on the last block. The decoder admits them in system turns only (any other role is a named
400). No dispatch wire carries them as sent: the gateway re-emits every mid-conversation system
turn as instruction text (user text on Anthropic rungs), and no other dialect has a positional
tool list. So every rung, Anthropic's included, receives one translation
(`exp/runtime/anthropic_protocol/tool_changes.py`):

| Change | Applied to `tools` | Stated in the system turn | Disclosure |
|---|---|---|---|
| addition by reference | the named tool loses `defer_loading` and is offered immediately | `Tool added: <name>. ...` | `messages.content.tool_addition->tools` |
| addition by value | the custom tool joins `tools`, replacing a same-named declaration in place | `Tool added: <name>. ...` (`Tool declared: ...` for a deferred definition) | `messages.content.tool_addition->tools` |
| removal | none: the tool stays declared, because earlier history may call it | `Tool removed: <name>. ...` | `messages.content.tool_removal->text` |

An added tool is therefore offered from the start of the request rather than only from its system
turn, a superset that keeps every tool the caller made available callable. Each change after the
first edits `tools`, so a rung's provider prompt cache for the tool prefix misses once per change,
where Anthropic's positional changes keep it. A cache marker on a block (or inside its
definition, never both) stays on that block's text line, the same prompt boundary.
`count_tokens` decodes the same way, so the count includes the added definitions.

A reference to a name that neither `tools` nor an earlier by-value definition declares is a named
400 on `messages.N.content.M.tool.name`, as on Anthropic (`tool_reference_unresolved`); a
definition reusing a declared server tool's name is a 400 on its `definition.name`. MCP references
(`mcp_tool_reference`, `mcp_toolset_reference`) and non-custom definitions (server tools,
`mcp_toolset`) are named 400s: declare the tool in `tools` and add it by reference.

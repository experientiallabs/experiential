# Responses input items without a canonical message

The Responses surface accepts five input item types, plus one variant of `function_call_output`,
that carry no canonical message of their own.
Each is valid OpenAI input whose meaning belongs to the provider that issued or interprets it, so the
decoder validates it shallowly (`exp/runtime/openai_protocol/responses_input_items.py`) and carries
the raw caller item at its position, so none of them falls into the message branch of the input
union (which would refuse it at `input.N.role`, a field the caller never meant to send).

| Item | Who sends it | Native OpenAI Responses rung | Any other wire |
| --- | --- | --- | --- |
| `agent_message` | Codex, for every inter-agent delivery (a sub-agent's task, a child's result); OpenAI's multi-agent beta item | Verbatim | A user turn: `Agent message from <author>:` followed by its plaintext parts. Encrypted and image parts drop as `input.agent_message.content.<kind>->dropped(unsupported_by_provider)` |
| `compaction_trigger` | Codex remote compaction (built-in OpenAI provider), always the final item | Verbatim; the provider answers with one `compaction` item | The rung narrows out of a mixed route; a route without a native Responses rung refuses before dispatch (below) |
| `context_compaction` | Codex, replaying compacted history an OpenAI backend issued | Verbatim | `input.context_compaction->dropped(unsupported_by_provider)` |
| `configuration_update` | Codex, a positional reasoning-effort change | Verbatim | `input.configuration_update->dropped(unsupported_by_provider)`; the request-level effort applies |
| `item_reference` | The AI SDK and LiteLLM for stored items, also as the SDK's typeless `{"id": ...}` | Verbatim, id kept even when it is not OpenAI-shaped; a dangling id is the caller's 400 | `input.item_reference->dropped(unsupported_by_provider)` |
| `function_call_output` without `call_id` | Codex 0.162+, a named output that answers no call (thread deliveries, notifications); openai-python 3.37 makes `call_id` optional | Verbatim | A user turn: `Tool <namespace>.<name> result:` followed by its text parts; other parts drop as `input.function_call_output.output.<kind>->dropped(unsupported_by_provider)` |

A `compaction_trigger` is not dropped on a foreign wire: the client accepts only a `compaction`
output item, so dropping the trigger would serve an ordinary, fully billed answer that Codex treats
as a fatal compaction failure and retries on the next turn. The refusal is `400`
`unsupported_parameter` on `input`, and its message names the remedy: compact client-side (Codex
compacts locally when the gateway is configured as a custom model provider,
`[model_providers.<name>]` with `wire_api = "responses"`), or use a model served on an OpenAI route.

The installed `openai` SDK schema does not enumerate `agent_message`, `context_compaction`, or
`configuration_update`, so the official-schema probe skips them, and it still requires a
`function_call_output` `call_id`, so the probe sees a placeholder; the strict wire models own
their shape. A malformed item of a known type is refused at its own field (`input.N.recipient`,
`input.N.reasoning.effort`), and an item whose `type` no model accepts is refused at
`input.N.type` with the whole accepted vocabulary.

A replayed `web_search_call` (a provider-executed search, forwarded verbatim to a native Responses
rung and dropped elsewhere as `input.web_search_call->dropped(unsupported_by_provider)`) is
accepted with the fields Codex serializes as optional left out: `id`, `status`, `action`, and a
`find_in_page` action's `url` or `pattern`, plus action types and statuses newer than the installed
SDK schema. Only the official-schema probe sees those fields completed; a field the caller did send
keeps its shape check. Because the official schema's item and action unions are not
discriminated, a branch whose own `type` refused the value is set aside while a sibling branch
still reports, so an invalid echo is named at its own field (`input.N.action.query`) and never at
another item's vocabulary (the computer-use and local-shell actions).

# Responses input items without a canonical message

The Responses surface accepts five input item types that carry no canonical message of their own.
Each is valid OpenAI input whose meaning belongs to the provider that issued or interprets it, so the
decoder validates it shallowly (`exp/runtime/openai_protocol/responses_input_items.py`) and carries
the raw caller item at its position. Before these models existed every one of them fell into the
message branch of the input union and the request was refused as
`Invalid value for 'input.N.role': expected a value.`, a field the caller never meant to send.

| Item | Who sends it | Native OpenAI Responses rung | Any other wire |
| --- | --- | --- | --- |
| `agent_message` | Codex, for every inter-agent delivery (a sub-agent's task, a child's result); OpenAI's multi-agent beta item | Verbatim | A user turn: `Agent message from <author>:` followed by its plaintext parts. Encrypted and image parts drop as `input.agent_message.content.<kind>->dropped(unsupported_by_provider)` |
| `compaction_trigger` | Codex remote compaction (built-in OpenAI provider), always the final item | Verbatim; the provider answers with one `compaction` item | The rung narrows out of a mixed route; a route without a native Responses rung refuses before dispatch (below) |
| `context_compaction` | Codex, replaying compacted history an OpenAI backend issued | Verbatim | `input.context_compaction->dropped(unsupported_by_provider)` |
| `configuration_update` | Codex, a positional reasoning-effort change | Verbatim | `input.configuration_update->dropped(unsupported_by_provider)`; the request-level effort applies |
| `item_reference` | The AI SDK and LiteLLM for stored items, also as the SDK's typeless `{"id": ...}` | Verbatim, id kept even when it is not OpenAI-shaped; a dangling id is the caller's 400 | `input.item_reference->dropped(unsupported_by_provider)` |

A `compaction_trigger` is not dropped on a foreign wire: the client accepts only a `compaction`
output item, so dropping the trigger would serve an ordinary, fully billed answer that Codex treats
as a fatal compaction failure and retries on the next turn. The refusal is `400`
`unsupported_parameter` on `input`, and its message names the remedy: compact client-side (Codex
compacts locally when the gateway is configured as a custom model provider,
`[model_providers.<name>]` with `wire_api = "responses"`), or use a model served on an OpenAI route.

The installed `openai` SDK schema does not enumerate `agent_message`, `context_compaction`, or
`configuration_update`, so the official-schema probe skips them and the strict wire models own
their shape. A malformed item of a known type is refused at its own field (`input.N.recipient`,
`input.N.reasoning.effort`), and an item whose `type` no model accepts is refused at
`input.N.type` with the whole accepted vocabulary.

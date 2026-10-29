# Replayed tool calls whose arguments are not one JSON object

Agent sessions resend every earlier assistant turn. Sometimes one of those turns
holds a tool call whose `arguments` text is not one JSON object. Two common causes:

- the model emitted malformed arguments (some self-hosted lanes do this);
- the call was cut off at `max_tokens`.

OpenAI's own API treats `arguments` as an opaque string and accepts this
history. The gateway used to refuse it with
`'messages.N.tool_calls.0.function.arguments' must encode one JSON object.` (Chat
Completions) or `'input.N.arguments' must encode one JSON object.` (Responses).
After one such turn, every later request in the session got the same 400, so the
session could not continue.

Now the gateway accepts such history on both surfaces. This applies only to history
turns. The current request's tool definitions and their schemas are validated as
before.

## What the decoder keeps

The caller's text is kept byte-exact. The decoder also derives one object to stand
in for it:

| Text | Stand-in object | Treatment |
|---|---|---|
| One JSON object | the object itself | none |
| A JSON string whose value encodes an object (double-encoded) | the inner object | `double_encoded_json` |
| A valid prefix of an object (cut off) | the prefix, closed | `truncated_json` |
| A JSON value that is not an object (`[1, 2]`, `"ls"`, `null`) | `{"_unparsed_arguments": "<exact text>"}` | `non_object_json` |
| Anything else (not JSON, two objects back to back, ...) | `{"_unparsed_arguments": "<exact text>"}` | `invalid_json` |

Blank text (`""` or whitespace only) is a zero-argument call. It is `{}` with no
treatment, as before.

Closing a cut-off object adds no content. It keeps every complete member and
element, and it keeps the characters of an unterminated string value. It drops
anything the cut may have shortened: a trailing number or literal (`12` may have
been `123`), a key with no value, a dangling comma, a partial escape, or half of a
surrogate pair. Text that is not a valid object prefix is wrapped, never repaired.

On Responses, the existing payload repairs (`input.arguments->serialized(json_object)`
and `input.arguments->decoded(double_encoded_json)`) run first, as before.

## What each wire receives

| Rung | Receives | Disclosed |
|---|---|---|
| OpenAI (`api.openai.com`) or Azure OpenAI, Chat or Responses | the caller's text, byte-exact | no |
| Any other Chat or Responses origin (OpenRouter, self-hosted vLLM, resellers) | the stand-in, encoded as one JSON object | yes |
| Anthropic Messages (`tool_use.input`) | the stand-in object | yes |
| Gemini (`functionCall.args`) | the stand-in object | yes |
| Bedrock Converse (`toolUse.input`) | the stand-in object | yes |

Non-OpenAI Chat servers parse history arguments to render a chat template (vLLM
calls `json.loads` on them). If the gateway forwarded raw text that is not an
object, the upstream would return the 400 instead. A provider 400 does not fail
over, so the gateway sends those servers the stand-in.

A rung that sends the stand-in adds one `ignored_parameters` entry for each
treatment present in the history. Chat Completions entries:

- `messages.tool_calls.function.arguments->decoded(double_encoded_json)`
- `messages.tool_calls.function.arguments->repaired(truncated_json)`
- `messages.tool_calls.function.arguments->wrapped(non_object_json)`
- `messages.tool_calls.function.arguments->wrapped(invalid_json)`

Responses entries use the same effects with the `input.arguments` path, for example
`input.arguments->wrapped(invalid_json)`.

Like other per-rung shaping disclosures, the admission reports the entries of every
rung in the route.

## Retained continuations

The in-process continuation store (`previous_response_id`) keeps the caller's text
with the turn. A store that serializes the conversation (`ContinuationState.model_dump`)
drops replay-only fields, including the raw text and its treatment. A turn read back
from such a store replays only its stand-in object, on every wire, OpenAI included.

## Code

- Decoding: `exp/runtime/openai_protocol/tool_argument_repairs.py`. The treatment
  is stored on `ToolCall.arguments_treatment`.
- Per-wire form and disclosures: `exp/runtime/models/providers/history_tool_arguments.py`.
  It is applied in `dialect_stream_payload` and disclosed per rung in `native_rungs`.

# Gateway reasoning display

Every gateway rung returns the readable reasoning its provider streams as display copy beside the
content. This page describes which reasoning is readable, how each public surface renders it, how
it interacts with replay and failover, and how an operator turns it off.

## What is displayed

Readable reasoning is OpenAI-compatible plaintext (`reasoning_content` / `reasoning`), Anthropic thinking, OpenAI
reasoning summaries and Gemini thought summaries, and the request builders ask for it where the
provider withholds it by default: Anthropic adaptive thinking carries `display: "summarized"`
unless the caller chose a display (a generation that thinks without a config gets that config
explicitly, unless the request carries sampling controls), Gemini thinking carries
`includeThoughts`, and a host-managed OpenAI Responses reasoning rung asks for
`reasoning.summary: "auto"` unless the caller chose a summary or the effort is `none` (a
customer's own key is never asked: OpenAI rejects summaries for unverified organizations). Chat renders it as `delta.reasoning` / `message.reasoning` (OpenRouter's
field, so `reasoning_content` keeps meaning the exposed plaintext or the sealed carrier), Messages
as one unsigned `thinking` block for non-Anthropic reasoning (Anthropic thinking keeps its own
signed blocks), and Responses as `summary_text` of a reasoning item with no encrypted content.
Display never changes replay: echoed display copy is caller-owned plaintext, forwarded only to
exposing rungs and dropped with disclosure elsewhere. A rung stamped `reasoning_output_hidden`
opts out, `EXP_GATEWAY_REASONING_DISPLAY=0` withholds it on every rung, and a request with an
output guardrail never displays reasoning, because the chain judges content it would not see. On such a rung the Responses surface also drops
provider summaries and projected thinking; the Messages surface keeps Anthropic's own signed
thinking blocks, which replay depends on.
Reasoning a caller did not see is retained once, as the capture's `provider_reasoning`, when the
collector's `capture_hidden_reasoning` is on; displayed reasoning is retained in the captured
response itself.

## Failover and timing

Readable reasoning that arrives before any other output is held privately, exactly like
route-bound reasoning: it refreshes generation timing but does not commit the attempt, so a
stall or an answerless stop still fails over to the next rung. The held text leads the
committed output once text or a tool call arrives, and later reasoning streams live. Anthropic
thinking and OpenAI reasoning summaries keep committing on arrival, as they always have.

## Exposure is separate

Exposure (`reasoning_output_exposed`, Tencent Hunyuan and DeepSeek think mode) decides whether a
rung replays caller plaintext `reasoning_content` and returns its plaintext in that field
instead of a sealed carrier; see gateway-architecture.md. Display never sets that field and never
replaces the carrier on a tool turn, so continuation pinning is unchanged.

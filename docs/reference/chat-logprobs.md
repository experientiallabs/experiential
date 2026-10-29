# Chat token probabilities

Chat Completions accepts `logprobs: true` and an optional integer `top_logprobs`
from 0 through 20. A count requires `logprobs: true`; zero is forwarded as zero,
and an omitted count stays omitted. `logprobs: false` remains a disclosed no-op.

Both streamed and aggregate responses use OpenAI's existing
`choices[0].logprobs.content` and `choices[0].logprobs.refusal` arrays. Records
preserve provider tokens, natural-log probabilities, nullable bytes and ordered
`top_logprobs` alternatives. The gateway does not tokenize or normalize them.
Probability-only chunks have an empty `delta`, and unrelated chunks carry null
logprobs. Null observations do not erase records already accumulated by a client.

A request selects only explicitly probability-capable `openai_compatible` rungs.
For a reasoning model, the deployment's `logprobs_reasoning_efforts` must also
include the effective effort, including the provider default when the caller
omits it. An empty declaration means unknown and rejects that reasoning rung.
Admission preserves route order, rejects when no eligible rung remains, and
rechecks the final per-rung request before freezing dispatch. It never changes
reasoning effort to make a probability request fit. A numeric `thinking_budget`
is not qualified by an effort declaration and cannot be combined with enabled
probabilities. Remove the budget and select a probability-capable effort, or
disable probabilities. This does not enable probabilities across provider
catalogs automatically.

Output guardrails and gateway-emulated stop sequences are unsupported with
probabilities because replacement or truncation can invalidate token alignment.
Input guardrails and provider-native stops keep their existing behavior.
Refusal probabilities follow the bounded precommit refusal buffer; an abandoned
attempt's probabilities never appear in the fallback answer. Content probability
records commit the serving attempt even when they precede visible text.

This support is limited to public Chat Completions on compatible Chat upstreams.
[Native Responses probabilities](responses-logprobs.md) use a separate opt-in
capability and request selector. Messages, batch and direct
`ModelRequest`/`ModelResponse` calls do not gain probability support. BYOK changes credential ownership, not model capability
or supported output format.

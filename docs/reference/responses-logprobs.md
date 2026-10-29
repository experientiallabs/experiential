# Responses output probabilities

Native OpenAI Responses requests may select `message.output_text.logprobs` and
an optional `top_logprobs` count. The selector and count remain independent
request intent: omitted, zero, and explicit values are preserved. The gateway
forwards the request fields without converting them to Chat `logprobs`.

The native parser preserves bounded provider records, including token bytes,
ordered alternatives, output item identity, content part identity, and the
phase in which each record was observed. Delta records are emitted once as
`response.output_text.delta`; cumulative `text.done`, `content_part.done`, and
`output_item.done` observations remain phase-specific. The terminal response
is authoritative for the final JSON snapshot. Empty `content_part.done`
probabilities are valid and do not erase other observations. The gateway never
synthesizes token bytes or performs tokenization.

Probability records are retained for replay, while the probability fields are
removed from the supported output-text message parts when constructing provider
history. Opaque tool and custom payloads are preserved. Failed attempts do not
commit their probability records, and a probability-bearing semantic event
commits the selected deployment before a later retryable provider failure.

Multipart probability content is rejected before encoding. Capability metadata
is native Responses specific and remains opt-in until deployment evidence is
reviewed. Chat probability behavior and non-Responses provider routes keep
their existing contracts.

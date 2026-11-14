# Gemini generateContent protocol

`exp/runtime/google_protocol` is the only Google Gemini API wire implementation, serving
`POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent?alt=sse` and
`:countTokens`, plus `GET /v1beta/models` and `GET /v1beta/models/{model}`, for callers of the
official `google-genai` SDK (point `http_options.base_url` at the gateway, `api_version`
`v1beta`) and raw REST callers. Callers authenticate with `x-goog-api-key` (the SDK default),
the `?key=` query parameter (Google's REST convention), or a standard Bearer header; all three
carry the same virtual key. Every failure on this surface is rendered in Google's error envelope
`{"error": {"code", "message", "status", "details"}}`: the gateway's stable machine code rides
`details[0].reason` (Google's `ErrorInfo` shape), the offending field `details[0].metadata.param`,
and a known wait a `RetryInfo` detail beside the `Retry-After` header.

## One serving abstraction

The body decodes to the same canonical `GatewayRequest` the OpenAI and Anthropic surfaces produce
(`surface = generate_content`), so every granted alias is reachable, not only Gemini models, and
the same authorization, routing, guardrails, capture, reservation and settlement apply. A request
routed to a Gemini or Vertex lane is re-encoded by the shared Gemini provider builder, so the
provider receives Google's own wire with the caller's `contents`, declarations, `toolConfig` and
`generationConfig` controls intact. A request routed to any other lane is translated like a
Messages request. The `{model}` is the catalog alias (an OpenRouter-style spelling such as
`<slug>:free` is admitted the same way as on `/v1`; the action is split on the last colon).

## Decoding

Both JSON spellings of every protobuf field are accepted (the SDK mixes `camelCase` objects with
`snake_case` keys such as `thinking_level` and `parameters_json_schema`). Supported and translated
losslessly: text, `inlineData` (PNG, JPEG, GIF, WebP, MP4-family video, WAV/MP3 audio, PDF) and
public http(s) `fileData` parts (image, video or PDF by declared type, inferred from the URL
when absent; any other declared type is refused), `functionCall` parts (with their `thoughtSignature` and Google-issued `id` carried onto the
canonical call so Gemini lanes replay both), `functionResponse` parts (paired by `id`, else by
name order as Gemini pairs them; a lone `result`/`content`/`output` string is the tool text,
anything else compact JSON), `systemInstruction`, `functionDeclarations` (`parametersJsonSchema`
or the upper-case `Schema` form, converted; Google's `nullable` becomes a JSON Schema `anyOf` with
`{"type": "null"}` beside the whole node, so sibling constraints keep applying to the non-null branch), `toolConfig.functionCallingConfig` (`AUTO`, `ANY`,
`NONE`, `allowedFunctionNames`; `VALIDATED` reads as `AUTO` with disclosure), and
`generationConfig` sampling (`temperature`, `topP`, `topK`; `presencePenalty` and `frequencyPenalty`
reach a lane only where its catalog says the model accepts them, else they drop with disclosure),
`maxOutputTokens`, `stopSequences`, structured output (`responseJsonSchema` or `responseSchema`
with `responseMimeType: application/json`; the MIME type alone selects schema-free JSON mode),
and `thinkingConfig` (`thinkingLevel` is the canonical reasoning effort reported under
`generationConfig.thinkingConfig.thinkingLevel`; `thinkingBudget` the numeric budget;
`includeThoughts` governs whether thought summaries are rendered).

Dropped with `x-experiential-ignored-parameters` disclosure: replayed `thought: true` parts (display
copy; a model turn of only thoughts drops whole and the history continues; a gateway-issued sealed
carrier echoed on a model turn replays authenticated instead), a
`thoughtSignature` on a text part, `safetySettings` (the gateway's own per-identity guardrails
apply), `labels`, `seed`, `responseLogprobs`/`logprobs`, `mediaResolution`, `audioTimestamp`,
`enableEnhancedCivicAnswers`, declaration `behavior` and a declaration's `response`/`responseJsonSchema`
(no gateway wire carries a function output schema). Refused with a field-specific 400:
`cachedContent` (a Google cache resource the gateway cannot address), Google server-executed tools
(`googleSearch`, `codeExecution`, `urlContext`, `fileSearch`, ...), `executableCode` and
`codeExecutionResult` parts, a part setting more than one `oneof` payload (`text` beside
`inlineData`), Google File API handles, `candidateCount` other than 1, non-text
`responseModalities`, `speechConfig` and `imageConfig`, and an unknown field anywhere. Parts keep
their order on the canonical message (an image, its caption, the next image). Pre-dispatch
rejections name Gemini's own fields (`contents`, `generationConfig.maxOutputTokens`, a rejected
sealed carrier as `contents.parts.text`), and a
request routed to an Anthropic Messages or Bedrock Converse rung receives the gateway's automatic
prompt-cache checkpoint exactly as a Chat Completions request does.

The cross-surface extensions ride the body as on every other surface: `provider: {"zdr": true}`
and the `gateway` request policy (through the SDK's `http_options.extra_body`).

`GET /v1beta/models/{model}` reads the whole segment as the alias (`coding:free` is one granted
spelling); only POST splits the trailing `:action`.

## Encoding

Streaming answers Google's SSE (`data: {GenerateContentResponse}` frames, one candidate): text parts
as they arrive (adjacent text merged per frame), each completed function call as one
`functionCall` part with its `id`, parsed `args` and `thoughtSignature`, generated images as
`inlineData`, and a terminal frame carrying `finishReason` (`STOP`, or `MAX_TOKENS` for an
output-capped turn) and `usageMetadata`. Thought summaries render as `thought: true` text parts
only when the caller set `includeThoughts`; a tool turn's hidden reasoning leaves as one trailing
opaque `thought: true` part holding the sealed carrier, which the next turn echoes verbatim. The
first frame carries `x-experiential-ignored-parameters` when anything was dropped, and every frame
the stable `responseId` and `modelVersion` (the alias). The non-streaming body is the same part
sequence in one `GenerateContentResponse`.

`usageMetadata` maps the normalized usage: `promptTokenCount` is the folded input the ledger bills
(cache reads inside it, reported again on `cachedContentTokenCount`), thinking is split out of the
output as `thoughtsTokenCount` with `candidatesTokenCount` the rest, and `totalTokenCount` covers
everything. Unknown usage renders the three mandatory counters as zero.

A provider refusal (a content-free `SAFETY`-class finish, an OpenAI refusal delta) has no part on
this surface and answers the shared 400 `refusal` error (`metadata.refusal_reason` carries the
bounded category), as on the Messages surface; mid-stream failures arrive as one Google error
envelope data frame, which the official SDK raises as its `APIError`.

`:countTokens` answers `{"totalTokens": N}` from the gateway's own reservation tokenizer, disclosed as
an estimate, without accepting, reserving or charging anything. `GET /v1beta/models` lists each
granted alias as `models/<alias>` with the three generation methods; an ungranted model answers
the shared no-oracle 404.

Google defines no idempotency header, so this surface never joins the keyed replay stores.

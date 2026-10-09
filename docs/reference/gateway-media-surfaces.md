# Gateway media surfaces

The gateway serves four OpenAI media APIs beside Chat Completions, Responses, and Messages. All
four are buffered (the caller receives one whole answer, never a stream), none has a replay
protocol (an inbound `Idempotency-Key` is ignored), and each is served only by aliases whose
catalog capabilities positively claim the surface on an OpenAI-wire connection. A chat alias
named on a media route, or a media alias named on the wrong media route, is refused with a 400
on `model` that names the route the alias does serve.

## Embeddings: `POST /v1/embeddings`

Served by aliases that declare `supports_embeddings`. Billed on the provider's reported
`prompt_tokens` with no output leg, and returned with the provider's exact vectors in `float` or
`base64` form. Input accepts a nonempty string, a batch of nonempty strings, a nonempty integer
token sequence, or a batch of nonempty token sequences. A flat token sequence produces one
vector; token IDs are forwarded unchanged and must match the selected model's tokenizer.
Boolean, fractional, negative, mixed-shape, and empty token inputs are rejected. Omit `stream`
or send the literal `false`; `true` and unknown parameters are rejected, and `stream` is never
forwarded. Reservations count token IDs directly and retain normal headroom; settlement uses the
provider-reported input usage. There is no response cache.

## Images: `POST /v1/images/generations`

Generations only: prompt in, images out. Served by aliases that declare
`supports_image_generation`, billed on the provider's reported prompt and image tokens, so a
model that answers without token usage is refused as unbillable rather than served for free.

## Speech: `POST /v1/audio/speech`

Text in (up to 4,096 characters, the OpenAI speech wire's cap), one audio file out. Served by
aliases whose gateway deployment capabilities declare `supports_speech`. `voice` is a built-in voice name or a custom voice
reference (`{"id": ...}`). The response is
the provider's audio bytes with the content type of the requested `response_format` (`mp3` by
default). `stream_format` is refused because the caller always receives the whole file. A
request whose worst-case audio (the slowest plausible reading, 4 characters per second in any
script, at the requested speed and format, plus SSE framing) would exceed the gateway's 64 MiB response buffer is refused on `input` before
any provider call; split the text, raise `speed`, or choose a compressed format.

Each lane bills one of two ways, chosen from its price card. A lane priced both ways (a unit card
beside token rates), or neither way, has no single meter to bill, so it is not served:

- A lane with a `character` unit card (`tts-1`) bills the input characters the gateway counted
  at that rate; the reservation is exactly the billed amount.
- A token-priced lane (`gpt-4o-mini-tts`) is asked for the provider's SSE format. The gateway
  reassembles the audio from the `speech.audio.delta` events and bills the token usage the
  `speech.audio.done` event reports. A stream without that usage event is refused as
  unbillable and fails over. Generated audio's length is not bounded by the text, so the lane
  must declare its provider-enforced `maximum_output_tokens`; that ceiling is reserved and
  enforced at settlement, and a token lane without it is not served.

## Transcription: `POST /v1/audio/transcriptions`

An audio upload in, a transcript out. Served by aliases whose gateway deployment capabilities
declare `supports_transcription`.

- The upload is a multipart `file` part (the official SDKs' shape) or a JSON `input_audio`
  object carrying base64 `data` and a `format`, up to 25 MB of audio either way. `source_url`
  is refused: the gateway never fetches a caller-named URL from its own network.
- The audio bytes stay in the data plane and are re-encoded as the provider's multipart upload
  on each attempt. Admission sees only the upload's size, filename, content type, SHA-256 (which
  binds the request digest to the recording), and the duration summed from the file's coded
  frames (Opus TOC bytes; AAC frames counted at HE-AAC's 2,048 samples, the larger profile; MP3
  and FLAC frame headers in their own containers; PCM sample counts in WAV; MP3, FLAC, or PCM
  wrapped in MP4 or Matroska is refused, since its timing would come from a sample table), never from container timestamps or headers, which
  the uploader controls. A container with more than one audio track is refused: a provider may
  decode any of them. An upload whose duration cannot be measured (a codec other than Opus, AAC,
  MP3, FLAC, or PCM, or a malformed file), or that measures longer than 4 hours, is refused with a 400 on
  `file`. The upload is read under a
  concurrency permit and the request deadline, like every other request.
- `json`, `verbose_json`, and `text` responses are served; `text` is rendered from the
  provider's `json` answer, which carries the meter. `timestamp_granularities` requires
  `verbose_json`. A JSON `input_audio.format` outside flac, m4a, mp3, mp4, mpeg, ogg, opus, wav,
  and webm is refused on that field. `srt`, `vtt`, `stream`,
  `chunking_strategy`, `include`, and the diarization reference fields are refused by name.
- A token-priced lane must declare its provider-enforced `maximum_output_tokens` (2,000 for
  `gpt-4o-transcribe`): a transcript's length is not bounded by the recording, so the reservation
  covers the audio input tokens plus that ceiling, and a token lane without one is not served.
- A lane with an `audio_second` unit card (`whisper-1`) bills the provider's metered seconds
  (`usage.seconds`, or `verbose_json`'s `duration`); a meter longer than the admitted duration
  is refused as inconsistent. A token-priced lane
  (`gpt-4o-transcribe`) bills `usage.input_tokens` and `usage.output_tokens`. An answer without
  the meter its lane bills on is refused as unbillable and fails over.
- The reservation covers the measured duration rounded up plus one second (never a
  caller-authored container header or an assumed bitrate, either of which can understate it)
  and is released to the observed cost at settlement.

A token-priced audio rung's provider usage must stay within the input and output tokens its
reservation covered (the gateway's estimate plus slack, and the declared output ceiling); usage
above either is refused as inconsistent and fails over. An alias whose deployment claims an audio
surface its wire has no endpoint for, or whose price card names no single billable meter (a unit
card, or both token rates with a declared output ceiling for transcription), is excluded at
startup.

## Per-unit pricing

Media priced by what it consumes or produces rather than by tokens carries a unit card on its
deployment's price card (`GatewayTokenPrices.units`): a unit kind (`character`,
`audio_second`, `video_second`, or `image`) and a map from variant to integer nano-USD per unit.
The empty variant is the flat rate; a named variant (`720p`, `1080p/audio`) prices one provider
SKU. The attempt freezes the card at reservation. Settlement prices the attempt's observed
billed units (thousandths of a unit, so fractional seconds stay exact, rounded half-up at one
nano-USD) at the rate of their variant, falling back to the flat rate, and adds any token cost.
A variant with neither its own nor a flat rate, or units of another kind, leaves the cost
unknown rather than invented. The card defaults to absent, so a token-priced deployment's
snapshot identity is unchanged by its existence.

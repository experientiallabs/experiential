# Gateway media surfaces

The gateway serves five OpenAI media APIs beside Chat Completions, Responses, and Messages.
Embeddings, images, speech, and transcription are buffered (the caller receives one whole answer,
never a stream); videos are asynchronous jobs the caller creates, polls, and downloads. None has a
replay protocol (an inbound `Idempotency-Key` is ignored), and each is served only by aliases whose
catalog capabilities positively claim the surface. A chat alias named on a media route, or a media
alias named on the wrong media route, is refused with a 400 on `model` that names the route the
alias does serve.

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
`supports_image_generation`. Each rung bills one of two ways, chosen from its price card:

- A lane with an `image` unit card (dall-e, xAI Grok Imagine) bills per image. The variant is the
  request's `quality/size` with omitted parts dropped (`hd/1024x1792`, `1024x1024`, or the empty
  flat variant); a card without that key prices the request at its flat rate. The reservation is
  `n` images at that rate, and settlement bills one unit per image returned. Token counts the
  provider reports are passed through to the caller and never billed. A unit card with no rate
  for the request's variant and no flat rate excludes the rung; when no rung remains the request is
  refused with a 400 on `size` before any provider call.
- Every other lane (the GPT image models) bills the provider's reported prompt and image tokens,
  so an answer without token usage is refused as unbillable rather than served for free.

An xAI connection is an `openai-compatible` connection on `https://api.x.ai/v1`. xAI takes
`aspect_ratio` and `resolution` rather than `size`, so callers omit `size` and `quality` on its
lanes (xAI answers a 400 the gateway relays otherwise). xAI refuses hosted (`url`) image output
for Zero Data Retention teams, so every request to xAI's own `/images/generations` asks for
`response_format: "b64_json"` whatever the caller sent, and the caller receives `b64_json` items
even when it asked for `url` (the gateway hosts no images to link to). Other providers receive the
caller's `response_format` unchanged. Per-image billing is the same either way. Published xAI
prices (2026-10-18): Grok Imagine Image $0.02, Imagine Image Quality $0.05, Imagine Image 2.0
$0.04 per image.

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

## Videos: `POST /v1/videos`

An asynchronous job surface shaped like OpenAI's Videos API, so the official SDKs'
`client.videos.create`, `retrieve`, `list`, `delete`, and `download_content` work unchanged. OpenAI
removed its own Videos API and every Sora model on 2026-09-24; the gateway serves the same contract
from Google Veo and xAI rungs. Aliases opt in with the gateway deployment capability
`supports_video_generation` and a `video_second` unit card.

- **Create** takes the SDK's multipart form or JSON: `model`, `prompt` (up to 4,000 characters),
  `seconds` (string or integer, default 4), `size` (default `720x1280`), and an optional
  `input_reference` first-frame image (JPEG, PNG, or WebP, up to 10 MiB) uploaded as a file part
  or sent as `{"image_url": "data:...;base64,..."}`. Remote image URLs and `file_id` are refused
  (the gateway never fetches a caller-named URL), as is `characters`. Sizes map to an aspect ratio
  and a resolution: `854x480`/`480x854` (480p), `1280x720`/`720x1280` (720p),
  `1920x1080`/`1080x1920` (1080p), `3840x2160`/`2160x3840` (4k). Sora's `1024x1792` and
  `1792x1024` have no provider equivalent and are refused on `size`.
- **Ids and ownership.** The gateway issues its own `video_<hex>` id and records it durably with
  the creating organization, identity, and key, the frozen deployment, the provider's job handle,
  and the ledger request and attempt carrying the reservation. Every read, download, and delete is
  scoped to the creating identity: another identity's lookup answers 404 exactly like an unknown
  id. The job record holds no prompt or image.
- **Failover happens only before acceptance.** Each rung reserves on its own ledger attempt and the
  job row is written before the provider sees the submit. A rung whose provider answers without
  creating a job (a throttle, credential, quota, routing, or server error, or a connection that
  never opened) settles at zero and the next rung serves; a 400 about the request itself is the
  caller's and is answered at once. A submit whose answer is lost may have created a job, so it is
  never re-sent: that attempt settles conservatively at its reservation and the caller gets an
  error. Once a provider accepts, the job stays on that rung.
- **Money.** The reservation is `seconds` at the lane's rate for the request's resolution variant
  (`720p`, `1080p`, `4k`, `480p`; a flat `""` rate prices every resolution). The accepted attempt
  holds it while the provider renders. Whichever process first observes a terminal state settles
  it exactly once: a completed video bills the rendered seconds (as the provider reports them,
  else the requested length) and never more than requested; a failed or safety-filtered render
  bills zero.
- **Polling and the settlement bound.** `GET /v1/videos/{id}` refreshes the job from its provider
  (at most one provider poll per job every 5 seconds), and every gateway worker runs a background
  poller that drives open jobs to settlement, so a job nobody polls still settles. A job still
  running 2 hours after acceptance is finalized as failed and its reservation released; the
  ledger deadline sits 10 minutes past that, so crash reconciliation never claims a job the poller
  still owns. A job whose provider dispatch was interrupted before its handle was recorded closes
  after 5 minutes at its reservation.
- **Content.** `GET /v1/videos/{id}/content` streams the MP4 from the provider through the data
  plane (`variant=video`; `thumbnail` and `spritesheet` are refused on `variant`, as no served
  provider renders them). A job that has not completed answers 400 `video_not_ready`; an asset past
  its provider's retention (Veo keeps videos for 2 days, reported as `expires_at`) answers 404.
  Provider credentials are sent only to the provider's own API host and never across a redirect.
- `GET /v1/videos` lists the caller's jobs newest first (`limit` up to 100, `after`, `order`), and
  `DELETE /v1/videos/{id}` hides a finished job (a running job answers 400 `video_in_progress`).

Providers (contracts and prices verified 2026-10-18):

- **Google Veo** on a `gemini` connection: `POST /v1beta/models/{model}:predictLongRunning`, then the
  returned operation is polled until `done` (`parameters.durationSeconds` is a JSON number;
  `aspectRatio` and `resolution` are strings). Veo renders 16:9 or 9:16 at 720p for 4, 6, or 8
  seconds, and 1080p or 4k for 8 seconds only; Veo 3.1 Lite renders no 4k. A first-frame image is
  sent inline. Paid-tier prices per second: Veo 3.1 $0.40 (720p, 1080p) and $0.60 (4k); Veo 3.1
  Fast $0.10 (720p), $0.12 (1080p), $0.30 (4k); Veo 3.1 Lite $0.05 (720p), $0.08 (1080p). A
  blocked render is not charged by Google.
- **xAI Grok Imagine** on an `openai-compatible` connection at `https://api.x.ai/v1`:
  `POST /videos/generations`, then `GET /videos/{request_id}` until `done`. Renders 1 to 15
  seconds at 480p, 720p, or 1080p; xAI reports the rendered duration (a JSON number, possibly
  fractional), which settlement bills in thousandths of a second. A first-frame image is refused
  on xAI lanes. Prices per second: `grok-imagine-video` $0.05, `grok-imagine-video-1.5` $0.08,
  `grok-imagine-video-1.5-lite` $0.02.

**Host output storage (xAI Zero Data Retention).** xAI refuses video generation for a Zero Data
Retention team unless the request names `output.upload_url`; xAI then `PUT`s the rendered MP4's
raw bytes to that URL and keeps no copy. A host enables this by installing an optional
`components.video_outputs` (`exp.runtime.gateway.video.output.VideoOutputStore`), and then every
xAI job takes this path whatever the team's retention setting (Veo has no upload option and is
unaffected). The store implements three methods, each safe to call concurrently from every worker:

- `upload_target(*, video_id, organization_id) -> VideoUploadTarget(url, expires_at)`: a presigned
  `https` URL accepting one `PUT` of the raw bytes with no extra headers, valid past the job's
  two-hour settlement bound. Called before submit; a failure settles the rung unsent at $0 and
  the walk fails over.
- `download(*, video_id, organization_id) -> VideoContentFetch | None`: how the data plane `GET`s
  the stored object (headers go to that URL only), or None when it is absent.
- `delete(*, video_id, organization_id) -> None`: removes the object when the owner deletes the
  job; a missing object is not an error, and a failure is logged without failing the delete.

A stored job completes on xAI's `done` without a provider URL, but only once `download` finds
the object: a `done` generation whose upload never arrived fails as `video_upload_missing` and
bills zero seconds. `GET /v1/videos/{id}/content` serves stored jobs from the store (503 when the
store is unavailable, 404 once the object is gone). Without a store, xAI requests carry no
`output`, and a Zero Data Retention team's provider 400 settles the attempt at $0 like any
submit refusal. `InMemoryVideoOutputStore` is a process-local reference implementation for
tests.

A deployment that claims video generation without a video job endpoint (any connection other than
Gemini or xAI's origin) or without a `video_second` unit card is excluded at startup. A gateway
composition without a video job store answers every video route with 404.

## ElevenLabs audio lanes

An `elevenlabs` connection (base URL `https://api.elevenlabs.io/v1`, key in the `xi-api-key`
header, default variable `ELEVENLABS_API_KEY`) serves the two audio surfaces above on ElevenLabs'
own wire; callers keep the OpenAI request and response shapes. ElevenLabs serves no chat, so a
completion naming an ElevenLabs alias is refused before dispatch.

- Speech posts `{text, model_id, voice_settings?}` to `/text-to-speech/{voice_id}`. `voice` is an
  ElevenLabs voice id (twenty alphanumerics) or one of the classic OpenAI names, mapped to
  premade voices: alloy to River, echo to Eric, fable to George, onyx to Brian, nova to Bella,
  shimmer to Sarah. `response_format` mp3, opus, pcm, and wav become `mp3_44100_128`,
  `opus_48000_64`, `pcm_24000`, and `wav_24000`; `speed` from 0.7 to 1.2 becomes
  `voice_settings.speed`.
- Transcription uploads the audio as `file` beside `model_id` (and `language_code` from
  `language`). ElevenLabs answers one JSON shape; the gateway renders `json` (the transcript
  plus a duration usage), `text`, and `verbose_json` (language, duration, and word timestamps;
  ElevenLabs reports no segments).
- An option the ElevenLabs wire cannot carry is refused on its own field when no rung of the
  alias can serve the request: an unknown voice name, `aac` or `flac`, a speed outside its
  range, `instructions`, a transcription `prompt` or `temperature`, or `segment` timestamps.
- Every ElevenLabs lane is unit-priced (a token-priced one is excluded at startup, since the wire
  reports no token usage): speech bills the input characters (`character`; $0.04 per thousand
  for Flash and Turbo, $0.08 for Multilingual v2, v3, and v4 on pay-as-you-go), and transcription
  bills `audio_duration_secs` (`audio_second`; Scribe v2 is $0.22 per hour), bounded by the
  admitted upload exactly like the OpenAI duration meter.

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

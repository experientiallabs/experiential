# Gateway capture hosting interface

Experiential owns one Rust collector for authenticated request context, HTTP
response capture, bounded lifecycle state and worker-owned delivery. Python
prepares the effective request during admission and configures a destination.
There is no Python callback per response chunk. Database work runs on the
destination worker; eligible response completion waits for its acknowledgement.

```python
from exp_gateway_native import CaptureCollector
from exp.runtime.gateway.native_capture import CaptureConfiguration, CaptureController

configuration = CaptureConfiguration()  # requires a hosted settlement decision
collector = CaptureCollector(configuration.model_dump_json(), write_record)
capture = CaptureController(collector, application_for=allowed_application)
# Pass capture to NativeControlPlane and collector to serve_native_gateway.
# Only after the final hosted lane decision:
collector.settle(request_id, keep_prompt=True, keep_response=served)
# A BYOK lane or denied policy retains nothing:
collector.settle(request_id, keep_prompt=False, keep_response=False)
```

`allowed_application(authorization)` returns an explicitly configured application
id or `None`. Organization and identity always come from authenticated authority,
never request metadata. The projection includes expanded messages, tool definitions,
generation settings and semantic provider context after input guardrails. Transport
replay keys and resolved provider credentials are excluded. Prompts may themselves
contain sensitive information; this is not content redaction or encryption.

Callers may send `X-Session-Id` with their actual harness session identifier.
Capture stores it as `request.context.session_id`, separate from the effective
provider request. It changes neither routing, authorization nor idempotency, and
is not forwarded to the provider. A single nonempty visible-ASCII value of at most
512 bytes is accepted; malformed or repeated values are ignored without rejecting
inference. No session is guessed from a prompt, cache key, user or request ID.
Other transport headers are never included in the capture document.

The synchronous `write_record(str)` destination runs on a dedicated Rust-owned
worker. New sinks validate schema2 with `CaptureRecord.model_validate_json`.
For persisted schema1 or schema2 documents, use `read_capture_record_json` from
`exp.runtime.gateway.native_capture`: it validates the selected version strictly
and upgrades schema1 with unknown `canonical_model_id=None`. The direct schema2
validator intentionally rejects schema1, and old strict schema1 consumers cannot
read schema2; update consumers together with the exact engine/native pair.
Unknown fields and versions remain errors. The effective request's `model_id`
retains its selected root identity. Record-level `canonical_model_id` names the
actual semantic winner, separately from the root and alongside `deployment_id`.
Both destination fields share the same output/permission retention boundary and
remain absent before that boundary. An accepted request that fails before routing
has no selected root model; a legacy record never gains an inferred winner.

A hosted, host-funded winner checkpoints its prompt before committed output becomes
client-visible. This preliminary schema2 record has no response, actual winner,
deployment, metrics or Gemini truncation assertion. The terminal update supplies
permitted response evidence and provenance later. BYOK winners do not checkpoint;
local collectors retain their completed-exchange flow. Destinations must recheck live
consent and merge updates idempotently, never allowing a late prompt-only update to
erase a stored response. Terminal denial alone cannot retract an earlier durable
checkpoint; the destination owns that live privacy enforcement. This is a required
hosted consumer behavior, not an append-only one-record-per-request contract.

An accepted checkpoint and any terminal update racing it retain the same request's
admission count and byte charge through their acknowledgements. Waiting updates use
the existing destination worker, with no thread or blocking task per request. The
request awaits a cancellable receipt bounded by its original deadline; cancellation
closes the provider transport and settles the attempt without waiting for storage.
The accepted capture jobs remain bounded and retryable after the caller leaves.
A close timeout reports unfinished capture rather than purging accepted content.
No-capture requests and local SQLite collection do not enter this checkpoint path.
The native request envelope defaults to 4 MiB; `CaptureController` no longer accepts
a separate `maximum_request_bytes` argument. Configure that bound on the collector.

Chat Completions, Responses and Messages HTTP surfaces share the same native tap.
JSON bodies and ordered SSE data payloads retain unknown fields. The observation
boundary is the native HTTP listener, which may feed a hosted relay; it does not
prove that an end user consumed every byte. `truncated` and
`client_disconnected` explicitly distinguish a prefix from complete evidence.
Output and settlement may arrive in either order. Keyed replays do not attach a
second response tap. WebSocket and batch response capture are not added here.

The winning rung's provider-returned plaintext reasoning is retained separately
in `provider_reasoning` when that rung explicitly permits reasoning exposure.
This preserves reasoning even when the public Responses representation carries
only an opaque continuation. Private provider reasoning is not decrypted for
capture. Capture permission never grants permission to expose hidden reasoning.
Reasoning is optional for every provider, including open models. Its absence never
rejects capture or inference; preserve returned evidence without inventing it.
Gemini thought parts use only the registered capture record's bounded evidence budget,
not inference output or refusal buffers. With capture disabled they are not retained.
`gemini_thought_parts_truncated=true` marks omitted over-budget evidence while inference
continues without an extra provider attempt. False means no truncation was observed;
null means unknown or not permitted, including schema1 reads. The marker is removed
with response-only evidence when retention denies that response.
`provider_tool_calls_json` retains completed calls as escaped JSON, including exact
argument text even when Messages presents the arguments as a parsed input object.
Chat tool turns on exposure-enabled routes return plaintext without appending
an opaque token to the same delta field; private routes retain authenticated tokens.

Postgres cannot represent NUL or lone UTF-16 surrogates. Affected request contexts
and responses include `source_json`, an escaped JSON string containing the exact
source value alongside the normalized query projection. Consumers recover the
request with `restore_capture_context`; response consumers decode `source_json`
when present. Reasoning uses `provider_reasoning_source_json` for the same case.
The escaped sidecars count toward all record limits. Oversize evidence is excluded,
not silently advertised as lossless. Historical reasoning stays in captured input
even when provider execution must omit it at a new user boundary.

The native queue carries structured request records and original response wire
buffers, not expanded response JSON trees. Records own immutable request context.
Only the destination worker parses response JSON, one record at a time.
JSON sizing counts fields and escapes without encoding response buffers; the
destination prepares its bounded payload once, before writing. A
hosted Python destination receives the same encoded string object on every retry;
storage retries do not decode the response or serialize the record again. Native destinations consume
the structure directly. Request admission still crosses the Python/Rust boundary
as JSON; exceptional lossless sidecars and raw tool-call strings also use JSON.

Hosted destinations can opt into `CaptureCollector.batched(config_json, write_batch)`.
The callback receives a tuple of prepared JSON strings and must return a list of
booleans in the same order: `True` acknowledges durable storage or an intentional
privacy exclusion, and `False` retains that record for retry. Exceptions or an
incorrect receipt count acknowledge nothing. Failed members retain the exact same
prepared string object; acknowledged members are released independently.
Once preparation succeeds, the redundant decoded record tree is released before
preparing the next member. Its admission charge remains until acknowledgement.
A preparation failure stops further decoding and batch gathering until that record
can prepare. It retains the sole decoded response workspace across retries; later
accepted records stay compact and charged in the queue. Already-prepared neighbors
still receive independent acknowledgements. This does not promise progress past an
unpreparable record or count the separately bounded decoded workspace as queue bytes.

Batches gather only already queued work, with no fill delay, up to 64 records.
Gathering stops after reaching a soft 1 MiB encoded-byte target; its final record
may cross that target, so the hard bound is 1 MiB plus the configured record limit.
The batch destination reserves five times that combined bound plus 256 bytes per
batch slot before queue admission. Rust rejects configurations without room for
this reservation and one queued record. Persistent failures can fill the bounded
batch or queue and backpressure new requests; batching does not promise unbounded
progress around failed records or process-crash durability.

Delivery limits bound record count, each final encoded payload and retained record
memory, including a record currently held by a slow destination. One destination
worker also reserves space for its prepared payload before any queue admission.
The Python destination reserves five times the encoded-payload limit plus 256 bytes:
one UTF-8 encoding, a worst-case four-byte Unicode string and object overhead.
Queued content uses the remaining budget, so a full queue cannot prevent preparing
its first record. The remainder must be at least the encoded-record ceiling;
configurations below that minimum are rejected in Python and Rust. Structured
records must also fit their charged heap budget; encoded length is not a heap-size
estimate. The UTF-8 encoder caps allocated capacity as well as payload length.
The reservation
is included in retained-byte counters while the worker prepares or retries a write.
String/vector capacity and conservative object-node charges are included; shared trees are charged
in every owning queue. The writer additionally owns one bounded response's decoded
tree while encoding and persisting it. Separate bounds cover in-flight entry count and memory, total response-buffer capacity
and request lifetime. Expiration runs on collector operations and once per second
on an idle destination worker; a blocked destination delays idle maintenance but
does not remove the memory caps. A response reserves its complete buffer allowance
before reading any provider bytes. Saturated delivery waits for capacity rather
than discarding records. Sustained storage pressure therefore increases latency and
limits throughput to what the destination can persist.

When capture is required but admission cannot register it, the gateway returns a
sanitized `capture_unavailable` 503 before provider dispatch. Policy-disabled capture
still serves normally. An eligible response does not finish successfully until its
destination write succeeds. A destination error retains the current record and
its queue slot, and retries with exponential backoff from 25 milliseconds to one
second. There is no retry-count expiry: a persistent outage backpressures capture
instead of discarding accepted data. A malformed or oversized payload that cannot
be prepared remains pending with a visible failure counter; raising a storage exception is never a way
to skip a record. Destinations must be idempotent because a write may commit before
its acknowledgement is lost. Returning normally also acknowledges an intentional
exclusion by the destination's current consent or retention policy.
Bytes already streamed cannot be withdrawn. Hosted eligibility can arrive after
the response ends, so hosts must also monitor destination failure counters for
these late writes. Destination exceptions never print potentially sensitive details.
`counts()` returns pending records, retained delivery bytes, successful destination calls,
failed preparation/write attempts, delivery drops and collector skips.
`maintenance_failures()` separately counts retention/WAL cleanup failures, including
maintenance during destination retries. A failed checkpoint does not make an
already committed record a failed write.
Failure counts do not count unique lost records: a recovered record has one success
and can have multiple failed attempts. A bounded `close()` drains
while releasing the GIL; a blocked destination cannot extend that caller's deadline.
A drain timeout returns false and leaves accepted delivery records queued, including
producers already waiting for space. It does not purge them. The host must keep the
process and destination alive to finish draining; closing its database before a
successful drain is unsafe. This memory queue is not a crash-recovery journal.
Unsettled hosted records remain pending until permission arrives or their TTL expires;
closing does not grant permission or purge them. Per-record size limits and explicit retention
policies still apply; this overload guarantee does not mean unlimited retention.

Destinations must enforce their own current consent, identity ownership, consent
generation, retention and physical storage constraints at the durable write. Native
admission policy is a performance gate, not a replacement for those checks. Capture
does not alter user-visible content, provider attribution, billing or the content-free
accounting ledger. The local CLI integration supplies the same collector with
identity/application bindings and a SQLite sink, without a hosted settlement gate.

The local sink captures completed Chat, Responses, and Messages exchanges, including
Messages SSE thinking signatures, tool blocks, and stop reasons. The stored exchange
also retains the original captured output frames and loss indicators. The canonical
trace exposes exact input/output messages, reasoning, raw tool arguments, and tool
error flags; tool failure is not evidence of whole-task failure or success.

Each exchange includes the full submitted or explicitly expanded history. Responses
parent links can recover earlier observed reasoning within the same identity. Missing
parents are labeled; unrelated Chat requests are never joined by matching prefixes.
Callers may optionally use request `metadata.conversation_id` to label an episode;
it takes precedence over `X-Session-Id`. Otherwise the captured session header
becomes the episode ID. Both remain scoped to authenticated identity and application.
Neither is required to capture a full submitted conversation.

Local retention enables SQLite secure deletion and checkpoints/truncates its WAL
after pruning. A concurrent reader can temporarily prevent WAL truncation; this is
reported as a maintenance failure and retried on idle maintenance, not claimed as
successful physical erasure. OS snapshots, backups, and storage-device remanence are
outside this database lifecycle. Expired rows are never returned by the reader API.

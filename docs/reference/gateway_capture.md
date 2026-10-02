# Gateway capture hosting interface

Experiential owns one Rust collector for authenticated request context, HTTP
response capture, bounded lifecycle state and worker-owned delivery. Python
prepares the effective request during admission and configures a destination.
There is no Python callback per response chunk. Database work runs on the
destination worker. Response completion waits for durable acknowledgement by
default. Hosts with an existing asynchronous lifecycle can explicitly opt into
queued delivery; accepted content remains owned until storage acknowledges it.

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
worker. Validate its input with `CaptureRecord.model_validate_json`. Schema version
1 includes the authenticated scope, effective request, optional response, model and
deployment provenance, and capture timestamp. The selected model is null for an
accepted request that failed before routing. A successful exchange emits one complete
record after both output completion and permission. A prompt-only permission emits
only the prompt. This avoids a delayed prompt update overwriting a full response.
A hosted collector can persist an early prompt checkpoint once the selected lane
permits capture. Final settlement independently gates the response. The destination
rechecks current privacy policy for every write, including delayed checkpoints.

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
`provider_tool_calls_json` retains completed calls as escaped JSON, including exact
argument text even when Messages presents the arguments as a parsed input object.
Chat tool turns on exposure-enabled routes return plaintext without appending
an opaque token to the same delta field; private routes retain authenticated tokens.

Postgres cannot represent NUL or lone UTF-16 surrogates. Affected request contexts
and responses include `source_json`, an escaped JSON string containing the exact
source value alongside the normalized query projection. Consumers recover the
request with `restore_capture_context`; response consumers decode `source_json`
when present. Reasoning uses `provider_reasoning_source_json` for the same case.
Native JSON numbers have a finite integer range. When a parsed value could have
rounded an integer beyond that range, capture preserves its original JSON using
the same `source_json` contract; wire requests use `body_source_json`. The source
is authoritative for exact numeric values, while the ordinary fields remain the
query projection. These sidecars do not change the inference representation.
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
The typed Python controller passes that JSON as immutable UTF-8 bytes through
`begin_bytes`, without decoding it into a Python Unicode string and converting
it back to UTF-8 at the native boundary. `begin(str)` remains available with the
same validation and capture contract.

Hosted destinations can opt into `CaptureCollector.batched(config_json, write_batch)`.
Batching does not change acknowledgement semantics. A host that already owns an
asynchronous capture lifecycle, such as Platform, may explicitly configure
`asynchronous_delivery=True`: checkpoint and terminal callbacks then return after
transferring already-admitted ownership, without waiting for a delivery slot or
durable storage. The same admission remains charged until storage acknowledges;
there is no unbounded overflow queue. Completed response bodies release unused
worst-case reservation while keeping their allocated bytes charged until decode.
The host must monitor
delivery failures and drain before closing its destination. Default and local
collectors continue to wait for durable acknowledgement. Neither mode adds a
process-crash recovery journal.

`truncate_request=True` is a hosted bounded-copy policy, not a serving mutation.
Oversized message content is trimmed largest-first, keeping as much text as fits
and never cutting below its 4 KiB prefix solely to make space. Every cut carries
an explicit byte-loss marker and capture-limit metadata. Equal-size candidates
retain their traversal order. Running encoded-size accounting avoids rescanning
the entire conversation for each cut. If even those prefixes cannot fit, the
existing explicit omission marker applies; capture is not advertised as lossless
outside its configured bounds. Local capture does not enable this policy by default.
The callback receives a tuple of prepared JSON strings and must return a list of
booleans in the same order: `True` acknowledges durable storage or an intentional
privacy exclusion, and `False` retains that record for retry. Exceptions or an
incorrect receipt count acknowledge nothing. Failed members retain the exact same
prepared string object; acknowledged members are released independently.
Destinations that accept UTF-8 can set `bytes_output=True` to receive immutable
bytes instead of strings. This skips Unicode decoding and re-encoding at the
storage boundary; the JSON, size limits and acknowledgement semantics are
identical. Failed members retain the same bytes object across retries.
Once preparation succeeds, the redundant decoded record tree is released before
preparing the next member. Its admission charge remains until acknowledgement.

By default, batched storage retains complete schema-1 records. A destination that
supports incremental updates can explicitly set `completion_references=True` to
receive schema-2 completions instead of repeating checkpointed request context.
A completion references a previously queued checkpoint: its `request`
contains `request_id`, `scope`, `protocol` and `model_id`, but no `context`.
All response, reasoning, usage and transport fields remain intact. The destination
must return `False` for a completion whose permitted prompt has not persisted yet,
and acknowledge intentional privacy exclusions. Complete-record and local SQLite
destinations retain the full schema-1 request/response contract.

Batches gather up to 64 records or wait up to 10 milliseconds from the start of
gathering. Gathering stops after reaching a soft 2 MiB encoded-byte target; its
final record may cross that target, so the hard bound is 2 MiB plus the configured
record limit. Shutdown flushes a partial batch without waiting to fill it.
The batch destination reserves five times that combined bound plus 256 bytes per
batch slot before queue admission. Rust rejects configurations without room for
this reservation and one queued record. Persistent failures can fill the bounded
batch or queue. Asynchronous handoff stays nonblocking while admission-owned
records and response memory fit their existing limits. Exhausting total admission
still refuses new capture, and exhausting response memory still backpressures
body collection. This is not unlimited outage tolerance or process-crash durability.

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
before reading any provider bytes. Synchronous delivery waits for capacity rather
than discarding records. Asynchronous delivery retains the handoff under its
existing admission charge instead. Sustained storage pressure can exhaust the
separate admission or response-memory bounds in either mode.

When capture is required but admission cannot register it, the gateway returns a
sanitized `capture_unavailable` 503 before provider dispatch. Policy-disabled capture
still serves normally. An eligible response waits for durable acknowledgement
unless its host explicitly enables asynchronous delivery. Asynchronous handoff
does not wait for delivery-queue capacity; it does not remove admission or
response-memory bounds. A destination error retains the current record and
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
Pending records and bytes include admission-owned handoffs waiting for a writer
slot, not only promoted delivery entries. These totals can exceed delivery-only
limits; the separate admission limits still bound waiting work. In-progress
requests that have not handed off are not delivery backlog.
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

# Gateway capture hosting interface

Experiential owns one Rust collector for authenticated request context, HTTP
response capture, bounded lifecycle state and asynchronous delivery. Python
prepares the effective request during admission and configures a destination.
There is no Python callback per response chunk and no database work on serving.

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

The synchronous `write_record(str)` destination runs on a dedicated Rust-owned
worker. Validate its input with `CaptureRecord.model_validate_json`. Schema version
1 includes the authenticated scope, effective request, optional response, model and
deployment provenance, and capture timestamp. The selected model is null for an
accepted request that failed before routing. It is an idempotent request update:
the response can arrive after an earlier prompt-only record. A hosted collector
does not enqueue content until terminal eligibility permits it, so queue overload
cannot lose a BYOK deletion behind an already queued prompt.

Chat Completions, Responses and Messages HTTP surfaces share the same native tap.
JSON bodies and ordered SSE data payloads retain unknown fields. The observation
boundary is the native HTTP listener, which may feed a hosted relay; it does not
prove that an end user consumed every byte. `truncated` and
`client_disconnected` explicitly distinguish a prefix from complete evidence.
Output and settlement may arrive in either order. Keyed replays do not attach a
second response tap. WebSocket and batch response capture are not added here.

Delivery limits bound record count, each encoded record and all queued string
capacity, including a record currently held by a slow destination. Separate bounds
cover in-flight entry count, encoded pending content, total response-buffer capacity
and request lifetime. Expiration runs on collector operations and once per second
on an idle destination worker; a blocked destination delays idle maintenance but
does not remove the memory caps. Saturation drops capture without delaying provider work.
Destination exceptions are counted without printing potentially sensitive details.
`counts()` returns pending records, retained delivery bytes, successful destination calls,
destination failures, delivery drops and collector skips. A bounded `close()` drains
while releasing the GIL; a blocked destination cannot extend that caller's deadline.

Destinations must enforce their own current consent, identity ownership, consent
generation, retention and physical storage constraints at the durable write. Native
admission policy is a performance gate, not a replacement for those checks. Capture
does not alter user-visible content, provider attribution, billing or the content-free
accounting ledger. The local CLI integration supplies the same collector with
identity/application bindings and a SQLite sink, without a hosted settlement gate.

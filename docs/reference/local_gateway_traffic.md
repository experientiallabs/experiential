# Build from local gateway traffic

## Shared engine contract

Local and hosted capture share Experiential's Rust `CaptureCollector`: one native
response tap, bounded input/output rendezvous and count/byte-bounded delivery worker.
Python supplies authenticated policy and the expanded post-input-guardrail context.
The local SQLite sink consumes those versioned records directly; it has no second
collector or queue. Tool definitions, generation settings and semantic provider
carriers are retained, excluding transport replay keys and resolved credentials.
The served request is unchanged. See [the hosting contract](gateway_capture.md).

## Local collection

Normal local gateway startup captures completed Chat Completions and Responses
exchanges for its authenticated identity grants. No hosted account or existing
router project is required. Traffic using your own provider keys is included.

```bash
exp run --root .exp
# Send ordinary requests with an issued gateway key.
# Stop the gateway; the bounded writer drains during graceful shutdown.
exp build support --source gateway --identity default --root .exp \
  --world-model world --judge judge --embedder embedder
```

The build reads `.exp/gateway/traffic.db`, separate from content-free accounting.
Use `--traces /absolute/path/traffic.db` to read another explicit capture file.
The identity is mandatory; omitting it never means all identities. The build
continues through the existing mining, world-model, evaluation and router
surfaces with their normal cost estimates and consent. Ingestion itself performs
no provider calls.

## Privacy and retention

`exp run --ghost` disables content collection while retaining content-free
accounting. The startup receipt reports `traffic_capture` and `traffic_database`.
The human-readable banner discloses capture before requests are served.
Turning capture off does not delete existing local data; remove the dedicated
traffic database only when no process is using it and you want that data deleted.

Native capture uses a bounded asynchronous writer, seven-day expiry, at most
10,000 records and 256 MiB of serialized payloads per identity, with a 1 MiB
per-record ceiling. SQLite indexes/journals add disk overhead. Oversize,
interrupted and non-successful responses are not reproducible completed records.
The collector's content-free counters report delivery and collection failures. Readers exclude
expired records even after the gateway stops.

Bindings reflect active identities and aliases at startup. Restart after changing
grants. Hosted consent and BYOK exclusion are separate Platform policy and are not
changed by these local defaults.

## Evidence, not inferred outcomes

Each record retains the request, function-tool definitions, generation settings,
post-guardrail expanded messages, provider-significant context and public response.
Transport authorization and resolved provider secrets are never copied. Prompt
content itself can contain sensitive information and is not automatically redacted.

The reader exposes observed assistant/tool-call/result sequences and retains the
original request/response context. A model completion is not proof of task success.
Missing context or unsupported output is excluded with a reason, not repaired.
Unrelated chats are not joined by matching their prompt text. The first bounded
page contains up to 1,000 retained records, the build's existing corpus ceiling.

This collection path covers JSON and SSE on Chat Completions and Responses.
Messages, WebSocket, batch, image and embedding traffic are not captured by this
local collector. An exchange is not a complete external-agent episode: tool
implementations and effects absent from subsequent traffic cannot be reconstructed.

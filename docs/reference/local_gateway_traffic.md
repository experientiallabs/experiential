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

Normal local gateway startup captures completed Chat Completions, Responses and Messages
exchanges for its authenticated identity grants. No hosted account or existing
router project is required. Traffic using your own provider keys is included.

```bash
exp run --root .exp
# Send ordinary requests with an issued gateway key.
# Stop the gateway; the bounded writer drains during graceful shutdown.
exp build support --source gateway --identity default --root .exp
```

Build reads and writes `.exp/gateway/traffic.db`, separate from content-free accounting.
Use `--traces /absolute/path/traffic.db` to read another explicit capture file.
The identity is mandatory; omitting it never means all identities. Build first preserves an
immutable normalized snapshot and project association, then mines scenarios and prepares their
world-model grounding. Source ingestion makes no provider calls; embedding work follows the
normal build cost estimate and consent.

Capture configuration, record contracts and the read-only store live under
`exp/runtime/gateway/`; trace normalization remains in `exp/common/traces/ingest/`.
The capture table contains observed exchanges only, with no training-token contracts,
adapter registry, model activation, or rollback machinery. This unreleased capture
format has no migration from earlier development databases. Preserve those files
separately and collect fresh traffic; startup rejects other database schemas rather
than silently hiding their records or modifying their contents.

## Privacy and retention

Capture is intended for a trusted local OS account whose gateway directory is not
writable by other users. It is not a shared-host tenant isolation boundary. The
full conversation is stored in plaintext for local replay and mining, including
any sensitive information inside prompts or tool output. No redaction, encryption,
or protection from another process running as the same account is implied. Use a
private account directory and appropriate disk/backup protection. On Windows, the
operator must restrict that directory with account ACLs.

On Unix, new content files use mode `0600`. Startup refuses group/other-accessible
existing databases or SQLite sidecars, and refuses symlink/non-regular content
files before writing. It does not silently change permissions or alter those files;
the operator must secure them first. This keeps the requested default-on collection
without weakening the stored evidence or claiming encrypted storage.

`exp run --ghost` disables content collection while retaining content-free
accounting. The startup receipt reports `traffic_capture` and `traffic_database`.
The human-readable banner discloses capture before requests are served.
Turning capture off does not delete existing local data. The shared traffic database also holds
explicitly imported evidence; removing it deletes those imports and project associations too.

Native capture uses a bounded writer with backpressure, seven-day expiry, at most
10,000 records and 256 MiB of serialized payloads per identity, with a 1 MiB
per-record ceiling. SQLite indexes/journals add disk overhead. Oversize,
interrupted and non-successful responses are not reproducible completed records.
When storage is slower than incoming traffic, response completion waits for the
SQLite transaction rather than discarding queued captures. Admission that cannot
register required capture returns 503 before making a provider call. A storage
failure retains the same prepared SQLite payload and retries with bounded backoff;
it does not re-encode the conversation or discard it after a retry limit. A payload
that cannot be prepared within its explicit bounds fails the HTTP body; bytes
already streamed cannot be withdrawn. Graceful shutdown reports an incomplete drain without purging accepted
delivery records, and the process must stay alive until that drain completes.
The collector's content-free counters report delivery and collection failures.
`maintenance_failures()` reports retention/WAL cleanup failures separately: a busy
reader cannot turn an already-committed capture into a reported write failure.
Cleanup retries during periodic maintenance. Readers exclude expired raw captures even after
the gateway stops. Explicit imports remain in their own tables and are not subject to capture
retention. Native delivery accepts the independently versioned import schema and never prunes it.

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
Unrelated chats are not joined by matching their prompt text. Ingestion reads the complete
retained identity snapshot in pages, with no total-record cutoff.

This collection path covers JSON and SSE on Chat Completions, Responses and Messages.
WebSocket, batch, image and embedding traffic are not captured by this
local collector. An exchange is not a complete external-agent episode: tool
implementations and effects absent from subsequent traffic cannot be reconstructed.

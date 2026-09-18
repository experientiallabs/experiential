# Gateway attribution and settled billing

The [gateway architecture](gateway-architecture.md) uses these contracts for caller
attribution and optional host-ledger billing on native completion surfaces.

## Settled billing extensions

An embedder can supply `NativeControlPlane(settled_billing_reader=...)` returning
`SettledRequestBilling` for a native-owned request ID. The reader supplies complete
settled request amounts across all attempts in nano-USD, never a current-price
estimate. It must be bounded and content-free. With no reader or no summary, billing
extensions are omitted rather than asserting zero. The native read never queues
behind busy bridge workers and waits at most 100ms or half the remaining request
lifetime, whichever is smaller. A timeout omits the annotation without delaying the
terminal until its delivery deadline; a timed-out callback retains its worker slot
until it finishes. At most one optional callback can execute per bridge, leaving
other workers for mandatory authorization and settlement. A one-worker bridge
skips optional enrichment entirely. Chat streams without a terminal usage event
skip the reader too.

The 100ms budget bounds waiting for a result, not Python execution. The host reader
must configure its own strict database/network deadline, shorter than the host's
shutdown budget. Native shutdown joins fixed bridge threads to release their
thread-local resources; an unfinished callback can therefore delay shutdown beyond
`graceful_timeout_seconds`. Readers must not block indefinitely. The engine does
not detach threads or kill Python work on timeout. Actual optional callback duration
is included in the bridge latency metric even when the HTTP caller stopped waiting.

Chat Completions, Responses, and Messages add `usage.cost` and `usage.is_byok` to
terminal responses; BYOK summaries also add
`usage.cost_details.upstream_inference_cost`. Existing token fields remain native
to each protocol. Streaming adds these only to the terminal usage event, never to
the Messages start-frame estimate. Keyed Chat/Responses store the annotated bytes
before publishing them, so replay returns exactly the original billing facts.
Messages continues to ignore `Idempotency-Key`. Empty and truncated completions
use the same billing projection. On Chat and Responses, if provider token usage is
unknown but settled money is known, usage contains billing only, never token zeros.
Chat streaming still requires `stream_options.include_usage=true` for that event.
Buffered streams read billing once and enrich only terminal frames.

The embedder's request ledger remains billing authority. These fields are a view of
that ledger, not an independent debit or a token-pricing implementation. USD JSON
numbers are display values: their binary floating-point representation does not
preserve every nano-dollar at multi-million-dollar request amounts. Exact
accounting and reconciliation use the host ledger's integer nano-USD amounts.

## Request attribution tags

Chat Completions, Responses, and Messages accept the optional `X-Explabs-Tags`
header. Use the official SDK's `extra_headers`; the body and provider metadata
stay unchanged:

```python
import json

completion = client.chat.completions.create(
    model="coding",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"X-Explabs-Tags": json.dumps({"team": "research", "env": "prod"})},
)
```

The same header works on `client.responses.create` and the Anthropic SDK's
`client.messages.create`, for streamed and non-streamed calls. Header names are
case-insensitive. Its value is one UTF-8 JSON object, at most 8 KiB on the wire,
with at most 16 unique string keys and string values:

- Keys match `[A-Za-z][A-Za-z0-9_.-]{0,63}`. Keys retain their exact spelling;
  `Team` and `team` are different. The `explabs.` prefix is reserved,
  case-insensitively, for platform-owned attribution.
- Values contain 1 through 256 Unicode scalar characters and no Unicode control
  characters. Do not send secrets, credentials, prompts, or personal information.
- Duplicate header fields, duplicate JSON keys (including escaped equivalents),
  nesting, non-string values, invalid UTF-8, and invalid keys or values return 400
  before admission or provider dispatch. Chat and Responses identify
  `param="X-Explabs-Tags"`, `code="invalid_parameter"`; Messages uses its native
  `invalid_request_error` envelope. Tag values are not echoed in errors.
- An absent header or `{}` means no tags. Tags are never forwarded as upstream
  headers, body fields, or provider metadata, and are not part of token estimates.
- A keyed Chat or Responses retry must carry the same tag map. JSON whitespace
  and key order do not matter; changing, adding, or dropping a tag returns 409
  `idempotency_conflict`. Messages remains unkeyed even with `Idempotency-Key`.
- Tags apply to this request only and are not inherited from prior turns. A
  Responses continuation may change or omit tags without changing its store's
  namespace. An injected hosted store retains its own organization, identity,
  and response-ID key; tags add no alias or revision binding.

### Host consumption contract

The native `admit` and `claim_scope` JSON callbacks carry a validated
`request_tags` map. The shared decoder revalidates it and places it on
`exp.runtime.gateway.contracts.GatewayRequest.request_tags`, typed by
`exp.runtime.gateway.request_tags.RequestTags`. An authority store's
`authorize_request` must copy that map onto `AuthorizationSnapshot.request_tags`
(the local SQLite authority does this). The synchronous ledger then receives it
through `accept_request(authorization=...)`, and each attempt's
`ExecutionSnapshot.authorization` carries the same attribution. Hosts own any
persistent tag indexes or analytics; this contract adds no SQLite tag columns.

Always compute the canonical digest with
`exp.runtime.gateway.replay_identity.canonical_request_sha256(request)`, not
plain request serialization: request tags are deliberately excluded from
`GatewayRequest.model_dump()` but explicitly included in keyed consistency.
`AuthorizationSnapshot.model_dump()` does include `request_tags`. Neither tag
field changes authority, routing, Responses retention, or the integer nano-USD
accounting unit. Both typed fields default to a fresh empty map.

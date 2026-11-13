# Native request transactions

A composition can supply `native_request_factory` to `NativeControlPlane`.
Admission still resolves the caller, route, provider capabilities and policies.
The factory then transfers that request's authority to a compiled request owner.
The native engine performs the serving work without entering Python callbacks:

- Select and reserve each physical attempt, including retries and fallback.
- Inspect input and output, including reinspection after a tool-search round.
- Rebuild provider messages, count tokens, and bind the new recovery identity.
- Sign the exact dispatch body and retain authenticated reasoning carriers.
- Create or reuse explicit caches under the host's frozen resource authority.
- Settle or release attempts and persist authorized Responses continuations.

Every serving request uses this contract. The local SQLite ledger supplies a
native owner automatically. Other compositions supply a compiled owner through
the factory. The synchronous Python ledger methods are administrative SDK calls
into Rust; they do not implement a second financial path. Missing native
authority prevents dispatch, and failed native operations remain with that owner.

Features that create paid cache resources require the compiled owner's cache
transaction capability. Admission-only cache plans cannot authorize a Python
resource-accounting callback. A composition without that native capability must
leave resource caching disabled.

## Ownership and recovery

`NativeRequestHandoff.registered` runs during admission, after registration
succeeds. Hosts use it to release admission-only memos or permits. Registration
failure leaves those memos with the admission failure path.

The native owner freezes terminal intent before starting durable I/O. A failed
write retains both intent and ownership for recovery. Replaying the same operation
cannot substitute a different terminal, release capacity twice, or report a
different charge. A usage report racing an acknowledged cancellation becomes
a retained `RECORD_LATE_METER` operation. The native ledger reconciles any allowed
provisional charge without reopening the request or duplicating dispatch and
health effects. Failure to persist that meter keeps the owner available for retry.

Disconnected terminals retain the provider's original nullable meter separately
from the engine's token estimate. Financial hosts can keep unresolved holds until
authoritative usage arrives without treating estimated tokens as provider evidence.

Shared health, load, affinity and observed-recovery state lives in Rust. Python
admission facades read or update that same state. Background configuration and
diagnostic consumers do not become request callbacks.

## Compiled host interface

`exp/runtime/gateway/native/include/native_callbacks.h` defines the versioned C ABI.
A host supplies a named capsule with a method mask and explicit retain, release
and invoke functions. Every selected method must own its complete contract,
including cancellation and recovery. All functions must be thread safe and must
neither unwind nor enter Python.

Inputs are borrowed only until invoke returns. The synchronous result sink accepts
one bounded result. The engine validates the table, method ownership and result;
an unavailable method or malformed result fails closed. The engine retains the
host context independently of the Python capsule for detached recovery work.

Methods 1-16 cover serving callbacks. Method 17 carries durable cache operations
between the native owner and its host. Method 18 preserves a late meter after
cancellation. Method 19 retains content-free inspection records asynchronously so
existing guardrails keep their observation and decision history after the accounting
owner takes over. These internal host operations are never HTTP endpoints.

## Payload and credential fidelity

Typed native request data preserves JSON integers beyond machine integer width.
Literal keys such as `$serde_json::private::Number` remain ordinary object keys.
Floating values retain the gateway's Python float semantics, and authenticated
history comparisons use the same numeric and Boolean equality as admission.
Strict decision parsing additionally rejects duplicate keys and non-finite floats.

Each signature covers the admission-bound endpoint and exact body digest. Explicit
credentials remain explicit; ambient AWS credentials use the native SDK provider
chain with shared, bounded, single-flight refresh. A refresh failure cannot reuse
an expired credential. Signing authority never enters response envelopes, capture
records or transaction receipts.

Inspection concurrency and compiled detector memory are bounded. Search rounds
rebuild the whole inspected conversation under the admitted provider capabilities;
they cannot append to an uninspected payload or broaden a signed route's authority.

Integration tests for host-owned model-chain authority use the optional Rust
`test-support` feature. Build the development environment with
`uv sync --extra dev --config-settings-package="exp-gateway-native:build-args=--features test-support"`
and run tests with `uv run --no-sync pytest`. The fixture validates issued receipts,
current grants and epochs inside the same native financial transaction. Normal
SQLite gateways still reject chain authority, including in this development build.
Release wheels omit this feature and its fixture constructor.

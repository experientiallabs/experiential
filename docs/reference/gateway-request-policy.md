# Per-request retries and routing

Chat Completions, Responses (HTTP and generating WebSocket frames), and Messages
accept the optional top-level `gateway` object. Official SDK callers send it through
`extra_body`. These controls belong to the gateway, never the provider payload.

```python
response = client.chat.completions.create(
    model="coding",
    messages=[{"role": "user", "content": "Summarize this change."}],
    extra_body={
        "gateway": {
            "retry": {
                "max_attempts_per_route": 2,
                "max_total_attempts": 3,
                "backoff": {
                    "type": "exponential",
                    "base_delay_ms": 500,
                    "max_delay_ms": 8000,
                    "multiplier": 2,
                },
            },
            "routing": {"allow_fallbacks": False},
        }
    },
)
```

## Bounds, not promises

- `max_attempts_per_route`: integer 1 through 4, including the initial generation
  dispatch. The ordinary retry ceiling is 2 when omitted. An explicit value caps
  **all** physical model calls on that route, including repaired reasoning replay
  and model turns following gateway tool search.
- `max_total_attempts`: integer 1 through 8, including initial dispatch, every
  retry, every fallback, reasoning repair, and tool-search continuation model turn.
  The server total ceiling is 8 when omitted.
- Omission of the per-route cap preserves the operator's ordinary retry mechanics
  and independently bounded semantic tool-search rounds. A semantic model turn is
  not an ordinary failure retry; it is still a physical dispatch and always spends
  the total budget. Explicit total 1 means one physical generation call, not one
  outer SDK operation with hidden redials.
- These are ceilings. Health circuits, failure eligibility, spending limits,
  authorization, context limits, and request deadlines can stop earlier. A larger
  retry ceiling cannot override any of them.
- Every model HTTP dispatch has its own durable reservation and settlement,
  including a reasoning-repair successor. A refused dial's observed usage belongs
  to its own attempt. Unknown usage stays unknown. Provider-side hosted tools are
  work inside one provider call; gateway tool-search model continuations are new
  calls. If the budget ends on a search-only turn, the gateway returns an explicit
  error rather than claiming an answer was produced.

## Wait modes

`backoff` is a closed discriminated object. No `default` mode exists.

- `{"type": "none"}` retries eligible ordinary failures immediately. It does not
  ignore a positive provider `Retry-After`: the gateway advances to an eligible
  fallback or returns the failure instead of hammering that route.
- `{"type": "exponential"}` starts at 500 ms, caps at 8000 ms, and grows by 2 by
  default. `base_delay_ms` is an integer 1 through 10000, `max_delay_ms` an integer
  1 through 60000 at least as large as the base, and `multiplier` a finite number
  1 through 4. Equal jitter selects a delay between half and all of the bounded
  exponential value. A provider retry floor is respected, never truncated to the
  ceiling. A wait must leave the next call its first-byte allowance within the
  original deadline.
- Omitted `backoff` preserves operator scheduling: ordinary retryable failures
  currently retry immediately, while throttles wait only under an authored
  throttle-redial policy. A caller's exponential mode changes the wait, not the
  eligibility of a generic 429. Reasoning repair and semantic tool rounds are not
  ordinary failure retries and do not receive discretionary exponential waits.

Only precommit work can retry or change route. Visible output freezes the serving
route. Cancellation during a wait prevents another generation dispatch. The native
HTTP client does not run an independent retry policy underneath accounting.

## Routing

`allow_fallbacks` is a strict boolean, default true. False keeps the first eligible
route after ordinary capability and governance filtering, and prevents runtime
fallback or saturation spill onto another route. This choice is frozen against the
caller's request before gateway-run search performs work. Injected search results
and later tool context are revalidated against that same route: if they no longer
fit, the gateway returns a field-specific context-window error without a model dispatch, even
if another route has a larger context window. Allow fallbacks or select a larger
eligible route when that flexibility is needed. Search preparation is not repeated
on another route after the no-fallback choice.

An optional `route_id` is an opaque public handle of the form `route_` followed by
64 lowercase hexadecimal characters. It is not a provider or deployment id. A host
that exposes handles resolves them within the caller's authorized model routes,
puts the requested route first, and may retain other eligible routes as fallbacks.
An unavailable, foreign, conditional-fallback-only, or incompatible selected route
is refused, never silently replaced during admission. Reasoning-pinned
continuations cannot be redirected to an incompatible route.

The host carries the requested handle in
`AuthorizationSnapshot.requested_route_id` and attests its resolution with
`GatewayRoute.resolved_route_id`. The engine checks both. The standalone catalog
resolver publishes no handles, so it fails closed when given a selector it cannot
resolve. Hosts must preserve the marker through route narrowing and reordering.

## Protocol and replay

All nested keys are closed and validated before dispatch. Booleans and numeric
strings are not integers. Unsupported surfaces, including embeddings, images,
decisions, Messages token counting, batches, and Responses WebSocket prewarm,
reject the extension rather than ignore it.

Effective caller controls participate in keyed replay identity; reusing a key with
changed policy is a conflict. Requests without the extension retain their ordinary
canonical hash. The extension is not prompt content and adds no reserved tokens.
Messages defines no idempotency key, and WebSocket upgrade headers do not assign
one key to every frame.

SDK retries are separate. Disable them (`max_retries=0` in the OpenAI or Anthropic
Python SDK) when an application needs to bound physical work across an entire
client operation, not merely within each gateway request.

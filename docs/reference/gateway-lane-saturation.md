# Lane saturation: the default in-flight bound and refuse-instead-of-overflow

A gateway worker admits at most `max_active_requests` requests at once (the data plane's
permit semaphore, default 64); a request past that waits for a permit until its own
deadline. On 2026-09-19 one organization sent ~100 requests a minute to a model whose lead
rung degraded to a two-minute first token. The rung authored no `concurrency_bound`, so
270–430 of its requests sat in flight, held every permit on every worker, and EVERY route
on the gateway — other models, `/v1/models` — waited minutes at the edge while CPU stayed at
30–60% and the CPU-keyed autoscaler never moved. `exp.runtime.gateway.lane_saturation`
closes that with two rules on top of the existing per-rung admission registry
(`exp.runtime.gateway.rung_admission`).

## The default lane bound

A rung that authors no `concurrency_bound` is bounded anyway, per worker, at
`default_lane_bound(max_active_requests)` = `ceil(max_active_requests * DEFAULT_LANE_SHARE)`
(half the permits: 32 of 64). Past it a reservation sheds sideways to the next rung exactly
like an authored bound (`dispatch_reason = queue_bound`). An authored `concurrency_bound`
replaces the default on its rung, higher or lower; authored rate windows on a rung with no
authored bound still get the default bound beside them. The host passes the bound when it
binds the control plane (`GatewayNativeBridge(..., default_lane_bound=...)`,
`NativeAttemptAccounting(..., default_lane_bound=...)`, `RungLoadRegistry(default_bound=...)`);
`None` leaves unauthored rungs unbounded, the historical behavior.

The share is per LANE, so a pool of two saturated lanes can still hold the whole worker
and a third lane would exceed it. The bound is protection against one slow lane, not a
per-pool budget; a pool that needs one authors `concurrency_bound` on each rung.

## Refuse instead of overflow

When every rung of a pool is at its bound the accounting used to force-admit the request
past the first shed rung and disclose `saturated_overflow` ("policy never manufactures a
failure"). Two things change:

- `GatewayRungDispatchPolicy.saturation` — `overflow` (default, unchanged) or `refuse`. An
  authored rung set to `refuse` answers the caller at once instead of dispatching one more
  request onto a lane already at its bound.
- A shed by the DEFAULT lane bound (`RungShed.default_bound`) refuses too: the default
  exists to protect the worker, so only a priority caller overflows it, and only to its
  level's ceiling (below).

The refusal is `lane_saturated_failure()`: failure class `throttled`, consumer-facing safe
message "This model is at capacity right now. Please retry in a few seconds. Pro subscribers
get priority access when models are busy.", `retry_after_seconds = 5`
(`THROTTLED_RETRY_AFTER_SECONDS`, the floor the protocol renderer applies to every throttled
wait, so the payload and the header agree).
The data plane renders it as the caller-facing 429 `unavailable_route` with `Retry-After: 5`,
before any dispatch, so the retry lands on a freed slot instead of queueing behind the slow
lane. Nothing is down, so it is not
`provider_internal`. The decision is `overflow_target(route, policy_sheds, shed_records)`; a
bypass that was not a registry shed (a cold throttle failover) keeps the historical overflow.
A reasoning-pinned continuation's first dispatch still force-admits its pinned rung for every
shed reason (`shed_keeps_pin`), the documented continuity-over-spill trade.

## Priority callers and default fairness

A PRIORITY caller (`AuthorizationSnapshot.priority_admission`: 1 for a paying organization,
2 for Pro, as the hosted platform sets it; 0 free) is not refused at a bound, on every rung
and with nothing to author: when either refusing bound (an authored `refuse` or the worker's
default lane bound) sheds it, `overflow_target` force-admits it onto the first bypassed rung
still below its ceiling (`saturated_overflow`), so on a saturated lane free callers get the
429 first. A priority caller's caller-selected first dial overflows its own rung the same way,
even on a soft (`overflow`) bound. The overflow is capped per level and floors to whole
slots (`priority_overflow_ceiling`):

| Bound | Paying | Pro |
| --- | --- | --- |
| Authored (`PRIORITY_OVERFLOW_FACTORS`) | 1.5x | 2x |
| Worker default (`DEFAULT_BOUND_OVERFLOW_FACTORS`) | 1.25x | 1.5x |

The default bound is already half the worker's permits, so its Pro factor stays below
`1 / DEFAULT_LANE_SHARE` and one lane's priority traffic never holds every permit. A rung whose
forced admission hit its ceiling (`RungShed.overflow_ceiling`) is skipped, and the request is
refused once every bypassed rung is capped.

Weighted fairness is ALWAYS on: every bounded rung, including one bounded only by the worker
default, weighs organizations by `AuthorizationSnapshot.fair_share_weight` under contention.
`GatewayRungDispatchPolicy.fair_share` stays in the contract only because persisted catalog
snapshots serialize it and their identity digests read it; its value no longer changes
admission (an authored `true` still requires a bound). No catalog digest moves.

## Durable retry and billing proof

A local capacity refusal carries `x-gateway-admission-refused: true` only after the
request's terminal ledger write commits a `failed_without_effects` certificate. Admission
must attest a direct target without web search or input-classifier checks, and the ledger
must verify that no provider attempt was recorded under the same serialized write lock.
Routed requests, requested searches and input checks remain uncertified because they can
incur charges before a model attempt. Historical and unspecified finishes default to false;
a later finish cannot upgrade an uncertified terminal record.
The same `Idempotency-Key` and request content can then retry safely: acceptance atomically
checks the prior certified failure and admits only one new owner. Insertion order identifies
the newest owner even when wall time moves backwards. Completed requests still
replay normally. Any recorded attempt, even one with zero reported cost, keeps the existing
replay barrier and conservative accounting. A failed terminal write never certifies free work.

`AttemptLedger.finish_request` and `SyncWriteLedger.finish_request` accept the trusted
`certify_no_effects=False` keyword and return the committed certificate. Custom ledger
adapters must persist it only on the first failed terminal transition, serialize its zero-attempt
check with attempt admission, and return it only after commit. HTTP relays strip upstream copies of the reserved
header; only their own durable admission authority can certify a local refusal. A generic
provider HTTP 429 is not proof of an unpaid request.

## Metrics

`rung_admission_counters()` returns `(sheds, saturated_overflows, saturation_refusals)`;
the observability snapshot and the text metrics expose `rung_saturation_refusals` beside
`rung_saturated_overflows`. A rising refusal count with a flat overflow count is a pool whose
every rung is at its default share on this worker: author a bound (or a second lane) for it.

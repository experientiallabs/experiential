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
- A shed by the DEFAULT lane bound (`RungShed.default_bound`) always refuses: the default
  exists to protect the worker, and overflowing it would protect nothing.

The refusal is `lane_saturated_failure()`: failure class `throttled`, safe message
"every lane for this model is at its in-flight bound on this gateway worker; retry in 2
seconds", `retry_after_seconds = 2`. The data plane renders it as the caller-facing 429
`unavailable_route` with `Retry-After: 2`, before any dispatch, so the retry lands on a freed
slot instead of queueing behind the slow lane. Nothing is down, so it is not
`provider_internal`. The decision is `overflow_target(route, policy_sheds, shed_records)`; a
bypass that was not a registry shed (a cold throttle failover) keeps the historical overflow.
A reasoning-pinned continuation's first dispatch still force-admits its pinned rung for every
shed reason (`shed_keeps_pin`), the documented continuity-over-spill trade.

## Metrics

`rung_admission_counters()` returns `(sheds, saturated_overflows, saturation_refusals)`;
the observability snapshot and the text metrics expose `rung_saturation_refusals` beside
`rung_saturated_overflows`. A rising refusal count with a flat overflow count is a pool whose
every rung is at its default share on this worker: author a bound (or a second lane) for it.

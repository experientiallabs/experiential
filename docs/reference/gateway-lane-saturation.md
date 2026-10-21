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

The refusal is `lane_saturated_failure(priority_admission)`: failure class `throttled`,
consumer-facing safe message "This model is at capacity right now. Please retry in a few
seconds." followed by a tier upsell (`LANE_SATURATED_UPSELL`): free callers get " Pro
subscribers get priority access when models are busy.", paying callers get " Pro orgs get even
higher priority when models are busy.", Pro callers get nothing more. `retry_after_seconds = 5`
(`THROTTLED_RETRY_AFTER_SECONDS`, the floor the protocol renderer applies to every throttled
wait, so the payload and the header agree).
The data plane renders it as the caller-facing 429 `unavailable_route` with `Retry-After: 5`,
before any dispatch, so the retry lands on a freed slot instead of queueing behind the slow
lane. Nothing is down, so it is not
`provider_internal`. The decision is `overflow_target(route, policy_sheds, shed_records)`; a
bypass that was not a registry shed (a cold throttle failover) keeps the historical overflow.
A reasoning-pinned continuation's first dispatch still force-admits its pinned rung for every
shed reason (`shed_keeps_pin`), the documented continuity-over-spill trade.
A conversation the host's cache placement names (`GatewayRoute.cache_placed_deployment_id`,
set by the hosted platform from its fleet-wide `gateway_cache_placements`) is kept on that rung
the same way on its first dispatch when the rung authors a `concurrency_bound` with
`saturation="refuse"`: a shed there never spills to a sibling rung holding none of its prompt
cache. A free caller gets the 429 above and retries onto the same warm rung; a priority caller
overflows a capacity shed on that rung up to its ceiling (a rate-window shed gives it the 429
too). A soft (`overflow`) or default bound keeps the historical sideways spill for placed
sessions. Failures on the rung are untouched: an operational failure (transport, timeout,
provider error, a circuit-open rung) advances the ladder, and a provider throttle follows the
pool's `failover_mode` and `throttle_cache_threshold` as before (`maximize_cache` surfaces it).
A request with no placement balances across the rungs exactly as before. On Experiential
Cloud's twin vLLM nodes a spilled turn recomputed its whole 100k-token prefix, and sheds caused
about two thirds of the fleet's uncached prefill.

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
| Authored (`PRIORITY_OVERFLOW_FACTORS`) | 1.25x | 1.5x |
| Worker default (`DEFAULT_BOUND_OVERFLOW_FACTORS`) | 1.25x | 1.5x |

A rung with an authored bound may author its own multiples
(`GatewayRungDispatchPolicy.priority_overflow_paying` / `priority_overflow_pro`, each in
`[1, 4]`); an unset one keeps the table's default, `1.0` turns that level's overflow off, and
the effective paying multiple is clamped to the effective Pro one. The worker default bound
never reads them. The hosted platform authors them per lane from its admin dispatch panel.

Admission order ends at the gateway unless the provider queues by priority too. A self-hosted
vLLM rung may author `GatewayRungDispatchPolicy.upstream_priority: true`; each dispatch to it
then carries vLLM's `priority` body field from the caller's level (Pro `0`, paying `1`, free
`2`; `native_rungs.UPSTREAM_PRIORITY`), so a server run with `--scheduling-policy priority`
schedules Pro ahead of paying ahead of free and preempts free first. Only the
`openai_compatible` and `openai_responses` wires carry it. Never author it on a third-party
rung: that provider would receive an unknown field. Unset (the default) sends nothing and adds
no identity bytes. Like every new dispatch field, author it only after every serving worker
runs a build that parses it.

## Request-class throttles (inert by default)

A host may vouch `AuthorizationSnapshot.priority_class`, a request class from `0` (free
account) to `8` (the top plan's paid request), plus optional class throttles. Every throttle
acts ONLY on a self-hosted rung: one whose dispatch policy authors `upstream_priority`
(`class_throttles.is_self_hosted_rung`, the same predicate that gates vLLM's `priority` field).
Third-party, BYOK and tenant rungs never author it, so on them every lever below is a no-op.

- **vLLM priority by class** (frees capacity). With a class, an opted-in rung gets vLLM
  priority `8 - priority_class` (`class_throttles.TOP_PRIORITY_CLASS`) instead of the
  Pro/paying/free map, so the server schedules and preempts by class.
- **Per-class in-flight cap** (frees capacity). `self_hosted_class_concurrency` caps how many
  dispatches of the request's class one worker holds on a self-hosted rung; past it the rung
  sheds as a hit overflow ceiling (never force-admitted), leaving GPU slots for other classes.
- **Output pace** (delivery only; the GPU still generates at full speed).
  `output_tokens_per_second` is evaluated at reservation (`class_throttles.output_pace`): with
  `output_pace_saturation` set, only while the rung's in-flight count is at or above that
  fraction of its bound (the authored `concurrency_bound`, else the worker's default lane
  share; an unbounded rung has no signal and is never threshold-paced). A paced attempt's
  `start_attempt` reply carries the rate and the data plane's Chat, Messages and Responses
  stream loops hold released events in a one-second token bucket over estimated tokens
  (`output_pacing.rs`), keeping heartbeats and never outliving the request deadline. An
  unpaced reply is byte-identical to before.
- **Class-ordered permits.** With `permit_priority_aging_seconds` in the serve config, requests
  waiting for one of the worker's `max_active_requests` permits are released by
  `priority_class` (carried on the admission), gaining one class per aging period so no class
  starves (`permit_queue.rs`). Absent, the permits stay the FIFO semaphore.

The class changes no admission bound, overflow factor or fair-share weight.

### The `:fast` request (`AuthorizationSnapshot.fast`)

A host sets `fast` for a request a caller asked to run faster at the priority card. Admission
asks for the priority processing tier (`class_throttles.fast_request`) and narrows to rungs that
carry a priority card; when none does it refuses (`require_fast_path`), never serving it at the
standard tier. On a third-party rung that is the provider's own priority tier, settled only on
witnessed evidence (a downgrade settles at standard). On a self-hosted rung (`upstream_priority`)
the tier is ours, so the request gets more of the GPU:

- vLLM priority `-1` (`FAST_UPSTREAM_PRIORITY`), ahead of every class, and no `service_tier`
  field on the vLLM wire;
- permit class `9` (`FAST_PRIORITY_CLASS`) in a configured permit queue;
- no class cap and no pace;
- a split bound: fast reservations may hold the upper half of the rung's bound (at least one),
  outside fair share; while any is held, every other reservation may hold only the lower half
  (at least one), hard, even for a priority overflow. Fast ignores the rung's total, so a full
  rung can briefly hold its bound plus the upper half until earlier requests finish. Like
  every registry bound this is per gateway worker, and rate windows still apply;
- or, on a rung that authors `fast_reserve_fraction` (0 < f < 1, needs `concurrency_bound`), an
  always-on reserve in place of the split: every other reservation holds at most the regular
  share, `(1 - f) * bound` rounded half up (at least one), whether or not a fast request is in
  flight, and never the full bound; a fast reservation admits below the bound. `0.25` holds others
  to 36 of 48 and 3 of 4; a bound too small to split (1, or 2 at 0.25) keeps no reserve. On a
  `refuse` lane both limits are hard, even for a priority overflow (a single-rung lane refuses Pro
  past the regular share). On an `overflow` lane they stay soft: a request past them spills to the
  next rung first, so fast has first claim on the kept slots, and admits on the forced pass exactly
  as an overflow bound does today, never a refusal (a forced pass, a forced rate redial included,
  is judged against the full bound). A forced fast request on an overflow lane is uncapped, like a
  free overflow, and counts toward a priority caller's first-dial overflow ceiling. Fair share (each organization's regular
  reservations only), the fresh-session threshold, trial headroom and pacing saturation of other
  traffic read the regular share. Under heavy regular load on a refuse lane fast has the reserve
  alone. Per worker, like the bound;
- settlement confirmed at the requested card (`tier_settlement`): our server reports no tier.

Every self-hosted lever needs the rung's priority card (`fast_on_rung`), so a fast request
never gets more of the GPU at the standard price. A caller's plain `service_tier=priority` that
admission keeps is treated as fast on the rung (`tier_authorization`): held and settled at the
card with these levers, except the permit class, which reads the host's authorization.

A fast request is placed by conversation like any other (`maximize_cache_affinity` keeps its
replica's prefix cache warm). A dedicated low-batch replica can author
`GatewayRungDispatchPolicy.max_prompt_tokens`: a prompt at least that long (lower-bound
estimate, prompt only) skips the rung at admission, after the window, capability and tier
narrowing, when an uncapped
rung of the same model stays on the route (`prompt_size.prompt_cap_indexes`), so long cold
prefills stay off it. The cap never refuses a request and never moves it to another model.

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

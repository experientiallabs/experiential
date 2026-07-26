# The cost/quality dial (`cost_quality`)

An endpoint fitted with `wmo optimize route fit --kind knn` routes each request to a pool model
using measured evidence, and keeps a pinned fallback for everything the evidence does not
support. The dial is the one number an operator turns afterwards to say how that endpoint should
spend: `cost_quality` in `[0, 1]`, where **0.0 spends for quality** and **1.0 spends as little as
this policy family can**. Nothing is refitted when it moves, so it is a live control.

`0.25` is the balanced position, and a fit left at the CLI defaults serves exactly those knobs, so
an endpoint nobody dials and an endpoint set to 0.25 behave identically. A fit given different
knobs (a stricter confidence bar, another novelty floor) is a different operating point: the dial
has not been set on it, the endpoint reports the coverage setting it was actually fitted with, and
no measured quality figure is quoted for it until someone dials it onto the frontier.

## What the dial actually changes

Two legs, in order. The balanced position sits between them.

| Dial range | What moves | Why it saves |
|---|---|---|
| `0.0` to `0.25` | The novelty floor opens from the 0.5 quantile of the fit set's own neighbor similarities down to 0.05 | The router is allowed to act on thinner coverage instead of abstaining to the fallback. The guard keeps its strict bar the whole way, which is why this leg costs almost no quality. |
| `0.25` to `1.0` | The cost knob ramps from 0 to 0.03, and the guard moves to its economic bar | Every candidate is priced before the pick, and a cheaper pick is accepted unless the paired evidence says it is significantly worse (rather than having to prove it is better). |

The second leg needs both halves. With the strict guard alone, a cost-tilted cheap pick fails
the bar and reverts to the pricier fallback, so turning the knob up *spends more*: measured on
RouterBench-derived tasks, cost bottoms out at -27.7% and is back to -15.1% by knob 0.08, having
given up 1.5 accuracy points on the way. The economic bar is what lets the cost knob act at all.

The confidence bar (`z`) is pinned to 0.5 at every dial position, because that is where every
anchor below was measured: one dial setting means one policy.

## Measured anchors

One held-out cohort (1199 scenarios, 9 models, 70/30 stratified splits, 5 seeds), quality and
cost both against the best single pool model on the same split:

| Dial | Name | Novelty floor | Cost knob | Guard | Quality | Cost |
|---|---|---|---|---|---|---|
| 0.00 | Quality max | 0.50 | 0 | strict | +1.14 pt | -13.9% |
| 0.25 | Balanced (default) | 0.05 | 0 | strict | +0.99 pt | -24.7% |
| 0.50 | Cost saver | 0.05 | 0.01 | economic | +0.87 pt | -40.8% |
| 0.75 | Deep saver | 0.05 | 0.02 | economic | +0.20 pt | -43.6% |
| 1.00 | Max savings | 0.05 | 0.03 | economic | -0.54 pt | -46.2% |

Those names are the labels the endpoint reports for each position, and they are display copy
rather than identifiers: read them, do not match on them.

Those five rows are measurements. **Every other dial position interpolates the knobs, not the
outcome.** Cost falls monotonically across the anchors and is expected to between them (both
knobs only ever make the router cheaper), but the curve is not a straight line, quality is not
monotone (the guard change at 0.25 measured cheaper *and* better on this cohort), and an
intermediate position is not a measured promise.

Two limits to keep in mind before quoting these numbers for your own traffic. The cohort is one
where the pool's models score within a few points of each other, which is the regime most
favorable to a cost knob; re-measure on your own outcome matrix before repeating a figure. And
the savings past 0.25 come with a weaker guard on the cheaper side: under a shuffled-label
control that configuration still cut cost by 38% while giving up only 0.15 points, which means
its savings do not depend on the neighbor evidence being informative. Treat a dial above 0.25 as
a decision to spend less, not as a free lunch.

Why the cost knob stops at 0.03: past it, measured cost turns around and rises while quality
keeps falling (0.05 → -45.1%, 0.08 → -38.3%, 0.12 → -31.2%), because the guard sends the
too-cheap picks back to the pricier fallback. A dial that ran past the turn would sell a worse
policy on both axes.

## Setting it

Three ways, same mapping.

**In Python**, as a pure transform of a fitted policy:

```python
from pathlib import Path

from wmo.optimize.knn import COST_QUALITY_ANCHORS, apply_cost_quality
from wmo.optimize.policy import RoutingPolicy

fitted = RoutingPolicy.load(Path("models/support/policy.json"))
cheaper = apply_cost_quality(fitted, 0.6)   # a copy; `fitted` is untouched
cheaper.save(Path("models/support/policy.json"))

for anchor in COST_QUALITY_ANCHORS:         # the measured table above
    print(anchor.cost_quality, anchor.quality_delta_points, anchor.cost_delta_percent)
```

The mapping is absolute: the knobs come from the dial alone, so re-applying any setting to an
already-dialed policy lands on the same artifact instead of compounding.

**On disk**, through the CLI, which also prints the anchor table:

```bash
uv run wmo optimize route tune models/support/policy.json --cost-quality 0.6
```

The first successful run copies the artifact to `policy.base.json` and every later run re-reads
*that*, so the dial is always applied to the policy as fitted. Tuning twice equals tuning once,
and sliding back down lands exactly where a first-time slide would. A tune that is rejected
writes nothing, and a snapshot left over from a superseded fit is refused rather than dialed
back over the current one: refit the policy and the command tells you to delete
`policy.base.json` first.

**Per served endpoint**, without touching the policy file, via `endpoint.toml` beside
`policy.json` in the model's directory:

```toml
# models/support/endpoint.toml
cost_quality = 0.6
```

The file is read at mount time. With no file, the policy is served exactly as fitted: mounting
never silently re-tunes an artifact.

## Reading and moving it on a live endpoint

Two routes, outside the OpenAI surface (`/v1/chat/completions` and `/v1/models` are unchanged):

```bash
curl -s localhost:8000/v1/endpoints/support/config
curl -s -X PUT localhost:8000/v1/endpoints/support/config \
  -H 'content-type: application/json' -d '{"cost_quality": 0.6}'
```

`GET` returns the current position, its label (one per measured position, per the table above,
with `Custom` for anything in between and `as-fitted` when no dial has been set), the knobs the
endpoint is actually serving, and the anchor table. Those knobs are read off the policy, so an
as-fitted endpoint reports the settings its own fit used; `floor_q` comes back `null` for a policy
fitted before that quantile was recorded, rather than a `0.0` that would read as "no floor". `PUT`
re-applies the mapping to the live runtime with no restart and persists the setting to
`endpoint.toml`, so it survives one. In-flight requests keep the position they started on.

`PUT` answers `409` when the policy has no dial (a `static` or `rank` endpoint) or when a
position past 0.25 is asked of a policy fitted without cost evidence, since there would be no
price for the cost knob to trade against. A position outside `[0, 1]`, or a non-finite one,
answers `400`.

## What the endpoint has actually saved

The dial says what an endpoint is trying to do; this says what it has done, totalled from the
endpoint's own request log (so a restart does not reset it):

```bash
curl -s localhost:8000/v1/endpoints/support/savings          # all time
curl -s "localhost:8000/v1/endpoints/support/savings?window=7d"
```

```json
{
  "requests_served": 6,
  "cost_saved_usd": 25.56,
  "cost_saved_pct": 94.67,
  "time_saved_s_estimate": 0.0,
  "expected_quality_delta_pt": -0.54,
  "estimate_basis": ["Savings compare what you were billed against ..."],
  "window": "all_time",
  "actual_cost_usd": 1.44,
  "baseline_cost_estimate_usd": 27.0
}
```

- **Cost** is logged dollars (`actual_cost_usd`) against the fallback priced on the same token
  counts (`baseline_cost_estimate_usd`). That counterfactual is an assumption: another model
  would have emitted a different number of output tokens and had its own prompt cache.
- **Time** is estimated against the median latency of this endpoint's *own* fallback-served
  requests, so it self-calibrates with traffic and is withheld until there are any. Differences
  are summed signed: a routed model that ran slower subtracts.
- **Quality** is the fitted expectation for the endpoint's dial position, carried over from the
  anchor table, never a live measurement. An endpoint tuned by hand to knobs off the dial gets no
  figure rather than a borrowed one.
- Savings can be **negative**, and are not clamped: an endpoint that routes toward a pricier
  model for quality says so.
- Every estimate names its basis in `estimate_basis`, and those sentences are meant to be shown
  to whoever reads the number. An endpoint with no traffic reports `requests_served: 0` with
  every other field zeroed.

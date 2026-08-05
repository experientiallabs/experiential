# Cookbook: RouterBench, routing on a precomputed matrix

The shortest path through the router: no world model, no trace capture, no closed-loop
sweep. When every candidate's outcome on every prompt has already been measured - which is
exactly what the RouterBench family of benchmarks publishes - the pipeline starts at `fit`.
Two commands take a precomputed outcome matrix to a fitted, dialable routing policy with a
held-out improvement report.

This is also the walk to read when you want to understand what the router actually learns,
because nothing else is in the way: one matrix in, one policy out, one report.

| Step | Command | Artifact |
|---|---|---|
| 1 | put an `OutcomeMatrix` JSON on disk | `matrix.json` (shape below) |
| 2 | `wmo optimize route fit matrix.json --kind knn --fallback <strongest>` | `policy.json` + `policy.json.bank.npz` |
| 3 | `wmo optimize route tune policy.json --cost-quality <0..1>` | the same policy, dial set |
| 4 | `wmo optimize route report matrix.json policy.json --baseline <anchor>` | `report.json`, held out |

The provenance rules from the [tau-bench walk](tau-bench.md#how-to-read-the-numbers-in-this-walk)
apply with two differences: a RouterBench-style matrix holds REAL completions at REAL
prices, so nothing below is world-model simulated; and because every row is a single-shot
completion, the report's cost-per-request IS its cost metric here - the cache-adjusted
effective-cost-per-completed-task machinery applies to served endpoints and multi-step
episodes, not to this matrix.

## Step 1: the matrix

The fitter consumes the same `OutcomeMatrix` JSON the sweep writes: a `pool` array (each
entry at least `name`, `kind`, `model`, plus its prices) and `outcomes` rows, one per
(scenario, model, episode), each carrying at least `scenario_id`, `task` (the prompt text),
`model`, `episode`, `reward`, `cost_usd`, and `call_seconds` (a list of per-call wall
seconds). A RouterBench-style dump maps onto it directly - one row per (prompt, model) with
the recorded correctness, cost, and latency. The published
`experiential-labs/wmo-routerbench-ours9` matrix is a complete worked example of the shape.

## Step 2: fit

```bash
uv run wmo optimize route fit matrix.json --kind knn --fallback gpt-5.5 \
  --out policy.json
```

The policy is a guarded nearest-neighbor lookup: the query embeds (the embedder
auto-resolves; `--embedder hashing` is the offline variant), its neighbors among the FIT
scenarios vote with their measured per-model rewards, and a non-fallback pick must beat the
fallback on paired per-neighbor evidence or the fallback serves. `--fallback` pins the model every request uses unless the paired
neighbor evidence clears the guard for a substitute - a statistical test on fit-set
neighbors, not a hard per-request quality bound; omitted, it defaults to the best single
model on the fit split. The fit reserves 30% of scenarios for reporting (a deterministic hash
split), and the fit-set accuracy it prints is labeled in-sample for exactly that reason.

## Step 3 and 4: dial, then report

```bash
uv run wmo optimize route tune policy.json --cost-quality 0.25   # balanced, the default
uv run wmo optimize route report matrix.json policy.json --baseline fable-5 --out report.json
```

The report replays the policy's serve-time selection over the held-out scenarios only (the
command output names the excluded fit count) and quotes all three objectives against the
baseline: quality, cost per run, latency p50/p95, plus the routed traffic mix.

## Measured results

Measured 2026-07-28 on `routerbench-ours9`: 1,199 RouterBench-derived prompts
(ARC-Challenge, HellaSwag, MMLU and siblings), each pre-executed against a 9-model pool
with real completions at real prices. Fit 839 scenarios, report on the 360 the fit never
saw. Anchor: fable-5 (the best-single check, gpt-5.5, follows).

| Fit | Traffic mix | Quality vs fable-5 | Cost vs fable-5 | p50 latency |
|---|---|---|---|---|
| Frontier-pinned (`--fallback gpt-5.5`) | gpt-5.5 38.1%, sonnet-5 32.8%, fable-5 23.1%, opus-4-8 5.0%, other 1.1% | 0.978 vs 0.897 (+8.1 pt) | $0.0024 vs $0.0033 (-29.4%) | 1.78s vs 3.23s (-45.1%) |
| Cost-leaning (fallback auto = sonnet-5) | sonnet-5 77.5%, fable-5 16.1%, gpt-5.5 3.6%, other 2.7% | 0.953 vs 0.897 (+5.6 pt) | $0.0012 vs $0.0033 (-64.0%) | 1.42s (-56.1%) |

Against the strongest single model in the pool instead of the anchor - the comparison an
expert will ask for, because fable-5 is dominated on this workload and makes the columns
above read generous - the frontier-pinned fit measures **0.978 vs gpt-5.5's 0.969
(+0.84 pt) at -33.8% cost** on the same 360 held-out scenarios. The original promotion gate
for this policy family (five seeded 70/30 splits, paired per seed) was re-run on 2026-07-28
and passed, reproducing its recorded per-seed numbers exactly: +1.04 pt mean over the best
single model at -26.6% cost, 5 of 5 seeds.

Concrete routed requests from the held-out set, with the guard's own reasoning verbatim:

- ARC science question -> **stayed on gpt-5.5**: "knn guard: reverted to gpt-5.5, evidence
  insufficient (105 paired neighbors, delta=+0.000, needs > 0.5xSE=0.012)".
- ARC electricity question -> **sonnet-5**: "knn: 151 neighbors, delta=+0.013 > 0.5xSE=0.010".
- HellaSwag continuation -> **fable-5**: "knn: 72 neighbors, delta=+0.070 > 0.5xSE=0.015".

Caveats that travel with the table: `ours9` is a RouterBench-DERIVED matrix over this
project's own 9-model pool, not the published RouterBench model grid, and the pool's models
score within a few points of each other - the regime most favorable to a cost knob. A
shuffled-label control still cut cost 38% past the balanced dial, so deep-dial savings do
not depend on the neighbor evidence being informative (see
[the dial reference](../reference/cost_quality_dial.md)). The quoted numbers are the
deterministic cached-vector protocol's (re-embedding live reproduces accuracy exactly and
moves cost/latency about 1% through embedding-API nondeterminism on near-tie neighbors).

## Reproduce it with one command

```bash
# in the research repo: github.com/experientiallabs/research
uv run reproduce run routerbench
```

Downloads the published matrix and recorded embedding vectors
(`experiential-labs/wmo-routerbench-ours9` on Hugging Face), refits the policy, rebuilds
both held-out reports offline - no credentials, no spend - and compares every number above
against the run, field by field, at bit-exact precision. Exit code 0 is REPRODUCED;
`verdict.json` carries the row-by-row comparison either way.

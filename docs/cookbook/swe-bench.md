# Cookbook: SWE-bench Verified, measured for real

Scope note up front: this page is deliberately a REPLAY page, not a pipeline walk. The grid
harness that bought these episodes drives the external SWE-bench harness and is not a product
CLI step, so there is no build/pool/optimize walk to retrace here; the canonical end-to-end
walk lives in [tau-bench.md](tau-bench.md). What this page owes you instead is the evidence
behind the shipped default and one command that reproduces it exactly.

| Step | Command | Artifact |
|---|---|---|
| 1 | `uv run reproduce run swe-bench` (research repo) | `verdict.json` + `policy.json` + `report_vs_fable-5.json` under the run's out dir |

The evidence behind the product's `swe-bench` default: 640 real mini-swe-agent episodes - 
20 pinned SWE-bench Verified instances x 16 candidate models x 2 episodes - each run inside
the instance's own SWE-bench Docker image and graded by the benchmark's own test suite
(FAIL_TO_PASS + PASS_TO_PASS; there is no LLM judge on this path). Provenance on every
number: `measured, real_episode`. Anchor: fable-5, zero unscored anchor cells, every
comparison at full paired n=20.

| Serving | Traffic mix | Quality vs anchor | Cost vs anchor | p50 latency per run |
|---|---|---|---|---|
| pinned opus-5 (the shipped default) | opus-5 100% | 0.850 vs 0.875 (−2.5 pt; CI −10..+5, unresolved) | $0.777 vs $1.220 per run (−36.3%); −34.4% per completed task (CI −46.5..−22.9) | 6.17s vs 8.04s |

The pin was chosen on the resolved cost CI, not a quality point estimate: at n=20 the only
quality difference this cohort resolves is qwen3.5-9b's collapse. The fitted kNN router
routes NOTHING at its balanced default on this workload (SWE-bench cost is turn-count
dominated, so model choice moves everything and re-routing 2 instances of 20 moves almost
nothing); every routed rung under leave-one-out refit is dominated by the pin.

The cautionary rows: qwen3.5-9b has the cheapest tokens in the pool and the HIGHEST cost
per completed task ($3.76, 2.7x the anchor - it solved 1 of 38 scored instances), and 15%
of cells hit the 75-step cap, concentrated in the cheap arms, so these numbers are
conservative for frontier candidates and harsh on cheap ones. The 75-step pin (vs
mini-swe-agent's shipped 250) means no solve rate here is claimable against a published
SWE-bench leaderboard.

## Reproduce it

One command, offline, credential-free - the shipped default is a static pin, so the replay
is arithmetic over the published outcome matrix:

```bash
# in the research repo: github.com/experientiallabs/research
uv run reproduce run swe-bench
```

Downloads the pinned matrix from
[`experiential-labs/wmo-swe-bench-defaults`](https://huggingface.co/datasets/experiential-labs/wmo-swe-bench-defaults)
(revision pinned in the manifest), replays the pin + report protocol, and says REPRODUCED
or DIVERGED against the published numbers above at bit-exactness.

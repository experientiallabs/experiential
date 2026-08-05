# Cookbook: terminal work, from a public corpus to Terminal-Bench 2.0

The same six steps as the [tau-bench walk](tau-bench.md), on a corpus anyone can download:
[`experiential-labs/wmh-terminal-tasks-traces`](https://huggingface.co/datasets/experiential-labs/wmh-terminal-tasks-traces)
on Hugging Face - 280 real bash-agent traces (1,370 OTel spans) of one-shot terminal work:
curl+jq registry lookups, filesystem and text processing, GitHub-via-curl, misc dev tasks.
Nothing in this walk needs data that is not public, which makes it the reproduction target
of choice: every command below runs from a fresh clone plus provider credentials.

One naming honesty note up front: this corpus is NOT Terminal-Bench 2.0. It is one-shot
terminal work, no multi-step engineering episodes, and no number measured on it is claimable
against any published Terminal-Bench leaderboard. The real Terminal-Bench 2.0 numbers now
exist separately and are at the [end of this page](#terminal-bench-20-measured-for-real);
the two are interpreted together there and never blended.

| Step | Command | Artifact |
|---|---|---|
| 0 | `just setup` | `.env`, synced dev environment |
| 1 | download the trace bundle, then `wmo build --file traces.otel.jsonl --name terminal-tasks` | `.wmo/models/terminal-tasks/` |
| 2 | `wmo providers set --pool-model ...` | `.wmo/pool.toml` |
| 3 | `wmo optimize model terminal-tasks` | matrix, policy + bank, held-out report |
| 4 | `wmo serve --name terminal-tasks` | the routed OpenAI-compatible endpoint |

Steps 0 and 2 are identical to the tau-bench walk, including its
[provenance rules](tau-bench.md#how-to-read-the-numbers-in-this-walk); read them there. The
distill and compression steps are optional here exactly as they are there; this page records
what they measured on this workload rather than re-walking them.

## Step 1: the corpus

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('experiential-labs/wmh-terminal-tasks-traces', repo_type='dataset',
                  local_dir='traces/terminal-tasks')"
uv run wmo build --file traces/terminal-tasks/traces.otel.jsonl --name terminal-tasks
```

The build cuts train/validation/test bands by trace identity before anything is fitted, so
every downstream held-out claim inherits a split that was fixed before any optimizer saw
the data. Measured build cost on this corpus class: single-digit dollars at medium fidelity.

## Measured results on this benchmark

Measured 2026-07-28, same evidence shape as the tau grid: 20 held-out test-band scenarios x
11 pool candidates x 2 episodes per compression arm, world-model simulated, judge named on
every artifact; policy fitted on 14 scenarios, reported on the 6 the fit never saw. Anchor:
fable-5. Provenance label on every number: `measured, wm_simulated`.

| Dial | Traffic mix | Quality vs anchor | Effective cost vs anchor | p50 latency |
|---|---|---|---|---|
| Every detent, 0 through 1 | sonnet-5 100% | -0.4 pt (paired CI touches zero: within noise) | $0.011 vs $0.035 (-69.4%, CI -77.3..-60.6) | -50.6% |

This workload's lesson is the constant policy. The evidence finds one model (sonnet-5) at
quality parity with the anchor at less than a third of the effective cost and half the
latency, and every cheaper candidate fails enough tasks that its cost per COMPLETED task is
higher, not lower. So the dial saturates immediately: max savings and balanced are the same
policy, and the router holds it at every detent. Best single model on the final matrix
resolves to kimi-k3 by raw mean (0.949 vs sonnet-5's 0.948) - a tie the router breaks
toward sonnet-5 on price.

Compression on this workload reproduces the inversion the compression track first measured
on financebench: the learned compressor arm RAISES effective cost per completed task for
strong models (fable-5 under it costs more than fable-5 raw), because dropped load-bearing
context fails tasks and lengthens episodes. Per-token accounting would have called that arm
cheaper; effective cost per completed task is the metric that catches it.

Caveats that travel with the table: 6 held-out scenarios is a small report band (wide cost
CI, quality read as within-noise rather than as a point estimate); the corpus is one-shot
bash work, so nothing here speaks to multi-turn terminal agents; world-model simulated
provenance throughout, with the judge (opus-4-8 under this project's build rubric) named on
the artifacts.

## Terminal-Bench 2.0, measured for real

Measured 2026-07-29 on the actual benchmark: 20 Terminal-Bench 2.0 registry tasks (pinned,
solve rates 0.15-0.92 on prior data, one deliberately-kept all-fail task) x 16 pool
candidates x 2 episodes = 640 real episodes under harbor's terminus-2 scaffold on E2B task
environments. The verifier is each task's own pytest suite - there is no LLM judge anywhere
on this path, so no judge-calibration caveat applies. Provenance on every number:
`measured, real_episode`. Anchor: fable-5, zero unscored cells, every arm at full paired
n=20.

| Serving | Traffic mix | Quality vs anchor | Cost vs anchor | p50 latency per run |
|---|---|---|---|---|
| pinned sonnet-5 (the shipped default) | sonnet-5 100% | 0.800 vs 0.775 (+2.5 pt; CI -15..+20, unresolved) | $0.946 vs $2.010 per run (-52.9%); -54.4% per completed task (CI -68.0..-30.2) | 6.14s vs 7.58s (-19.0%) |

The fitter agrees with the pin: given the full matrix it selects sonnet-5 as its own
fallback and routes away from it 0.0% of the time, and under leave-one-out refit the best
routed rung (-45.9% cost at -2.5 pt) is dominated by simply serving sonnet-5. Other honest
operating points from the same matrix: gpt-5.6-sol at 0.700 for $0.46 per completed task
(-82.2%), and qwen3.6-27b at 0.550 for $0.16 (-93.7%, a real quality trade at -22.5 pt).
The cautionary rows reproduce too: deepseek-v4-pro costs +121.5% per completed task
(content-filter churn), and qwen3.5-9b, the cheapest tokens in the pool, solves 0.10.

### Read together with the terminal-tasks table above

Two independent legs reach the same policy. The simulated terminal-tasks run said "constant
policy, sonnet-5 at every detent, quality within noise at -69.4% cost"; the real
Terminal-Bench 2.0 grid says the same constant policy, with the stronger point estimate
(+2.5 pt ABOVE the anchor) at -52.9% per run. That agreement is aggregate-level only, and
the numbers must never be blended:

- The provenances differ (`wm_simulated` vs `real_episode`) and the corpora differ
  (one-shot bash vs multi-step engineering episodes).
- Per-scenario transfer between a simulated terminal leg and real TB2 measured NEGATIVE
  (Spearman rho -0.162 on an earlier cohort): the simulation picks the right MODEL while
  ranking scenarios no better than chance, so only the aggregate conclusion carries.
- Latency does not reproduce as stated: the simulated leg's -50.6% reads -19.0% p50 per
  run on real episodes (and a per-completed-task cut of the same data shows sonnet-5
  SLOWER than the anchor at +33.5%, because it keeps working on tasks the anchor gives up
  on). Cost and quality survive contact with the real benchmark; the latency claim
  becomes definition-dependent and should be quoted per run with the definition named.

The product outcome: the platform's `terminal-bench-2` default serves pinned sonnet-5 with
this table installed as its evidence, and the quality claim ships as "+2.5 pt, unresolved
at n=20" - never "beats the anchor" - because a 20-scenario paired CI cannot resolve
single-digit differences in either direction.

### Reproduce it

One command, offline, credential-free - the shipped default is a static pin, so the replay
is arithmetic over the published outcome matrix:

```bash
# in the research repo: github.com/experientiallabs/research
uv run reproduce run terminal-bench-2
```

Downloads the pinned matrix from
[`experiential-labs/wmo-terminal-bench-2`](https://huggingface.co/datasets/experiential-labs/wmo-terminal-bench-2)
(revision pinned in the manifest), replays the pin + report protocol, and says REPRODUCED
or DIVERGED against the table above at bit-exactness.

# Cookbook: terminal-tasks, the fully public end-to-end run

The same six steps as the [tau-bench walk](tau-bench.md), on a corpus anyone can download:
[`experiential-labs/wmh-terminal-tasks-traces`](https://huggingface.co/datasets/experiential-labs/wmh-terminal-tasks-traces)
on Hugging Face - 280 real bash-agent traces (1,370 OTel spans) of one-shot terminal work:
curl+jq registry lookups, filesystem and text processing, GitHub-via-curl, misc dev tasks.
Nothing in this walk needs data that is not public, which makes it the reproduction target
of choice: every command below runs from a fresh clone plus provider credentials.

One naming honesty note up front: this corpus is NOT Terminal-Bench 2. It is one-shot
terminal work, no multi-step engineering episodes, and no number measured here is claimable
against any published Terminal-Bench leaderboard.

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
quality parity with the anchor at roughly a quarter of the effective cost and half the
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

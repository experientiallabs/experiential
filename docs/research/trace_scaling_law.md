# Trace scaling law

**Does feeding the world model more recorded traces improve how faithfully it reconstructs the
environment?** We sweep the number of *training* traces against a fixed held-out test set and measure
open-loop reconstruction fidelity (RAG-only — the shipped base prompt with a retrieval buffer over
the training traces, no GEPA). One curve per benchmark, anchored at `n=0` (the no-RAG baseline: the
same base prompt with retrieval turned off).

The corpus is split deterministically (hash of `trace_id`) into a fixed `test` band (the y-axis,
never changes as the corpus grows), a fixed `valid` band, and a train *pool*; each sweep point draws
`n` traces from the pool, shuffled per seed and nested as `n` grows, so the only thing varying along
x is how much training data the model sees. `wmh/research/trace_scaling.py` implements the ablation.

![Trace scaling law](trace_scaling_law.png)

## The finding: for two of three benchmarks, trace data barely contributes

The `n=0` (no-RAG) anchor is the key: since RAG is the only channel through which trace data reaches
the world model here, the gap from `n=0` to the RAG curve *is* the entire contribution of the
training traces. That gap is large for tau-bench and near-zero for the other two:

| Benchmark | n=0 (no-RAG) | n=1 | n=4 | largest | RAG lift (n=0 → max) | corpus (traces / steps) |
|---|---|---|---|---|---|---|
| tau-bench (tool calls) | 0.641 | 0.844 | 0.887 | 0.932 (n=648) | **+0.29** | 1033 / 5289 |
| terminal-tasks (bash) | 0.854 | 0.860 | 0.858 | 0.873 (n=164) | +0.02 | 280 / 685 |
| swe-bench (arbitrary code output) | 0.725 | 0.726 | 0.729 | 0.743 (n=173) | +0.02 | 255 / 1868 |

For **terminal-tasks and swe-bench**, zero traces already scores 0.854 / 0.725, and the full pool
moves it by ~0.02 — trace data adds almost nothing, and what little there is arrives with the first
neighbour. Their fidelity is essentially the base model's zero-shot competence, not anything learned
from the corpus. For **tau-bench**, retrieval is the whole story: +0.20 by the first trace and +0.29
by the full pool, still climbing at n=648 (it has not plateaued). The across-seed std is ≤0.014 at
every point except tau-bench's `n=1` (0.028), so both the tau climb and the flatness elsewhere are
signal, not noise (and each seed draws a different shuffled subset, so it is not a lucky sample).

**Why the split?** Retrieval keys on the `(state, action)` of each step. In tau-bench the observation
is a deterministic lookup of state the arguments name (a user record, a reservation), the same
arguments recur across traces, and a bigger pool increasingly contains a near-duplicate whose
observation is essentially the answer — so more traces keep helping. In terminal-tasks and swe-bench
the observation is a function of environment state the trace never captures (live web responses, the
repo's file contents) and is near-unique per step; retrieval finds a lexically similar *command* but
its *output* is unrelated, so no number of traces makes it predictive. (Empirically: the retrieved
demo's observation resembles the actual one with a large lift over random for tau at high `n`, but
~zero lift for terminal/swe. See `.agents/docs/research/` for the diagnostic.)

**Takeaway:** more traces of the same kind only buy fidelity when the observation is a retrievable
function of the recorded `(state, action)` — true for structured tool lookups (tau), false for
outputs that depend on uncaptured environment state (shell, code execution). For the latter the
leverage is elsewhere: richer state capture in the trace format, and prompt/optimization.

**This is reconstruction, not memorization.** A natural worry is that the model only *looks* like a
world model because it pretrained on these public benchmarks. A contamination probe refutes it:
scored **zero-shot** — no retrieved examples, no trajectory history, just the action — the model
reproduces almost nothing (swe-bench, the most public via GitHub repos, is *lowest*: 0.11 factuality,
3% exact match; tau-bench, synthetic and un-memorizable, is the 0%/0.12 control). Verbatim
memorization would show near-1.0 factuality. Adding the recorded trajectory history roughly *doubles*
factuality (swe 0.11→0.40, terminal 0.27→0.63) — the model reconstructs from in-context signal, which
is exactly why the n=0→RAG gap above measures a real contribution rather than recall. Probe script
and raw numbers: `.agents/docs/research/rag_opt_results/` (`contamination_probe.py`, `contam.log`).

## Reproduce

The curves come from `run_trace_scaling.py` (RAG-only, `--modes base`) — a workspace script,
snapshotted below as of publication (`.agents/` contents are disposable; the commands quoted
here are the record) — scored with the
canonical `RubricJudge` on a fixed test split, parallelized (`--concurrency`) and cost-bounded
(`--test-cap`). Raw `AblationReport` JSONs were archived to the workspace
(`.agents/docs/research/trace_scaling_results/` as of publication).

> **Judge provenance.** These numbers predate #83 and use **rubric-v1** (unweighted mean of five
> dimensions). Main's **rubric-v2** (factuality-weighted headline + validity flag + middle
> truncation) scores strictly lower on identical predictions — ≈0.58 where v1 scored ≈0.70 — so
> re-running the commands below on `main` will produce lower absolute fidelities. The *shape* of each
> curve and the n=0→RAG gaps (what this doc claims) are unchanged in kind.

```bash
# one benchmark's curve — the same --counts for all three; the ablation auto-caps at the train pool
# (tau-bench -> 648, terminal-tasks -> 164, swe-bench -> 173), so the top point is the whole pool.
AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_trace_scaling.py \
  terminal-tasks --counts 1,4,16,64,256,648 --modes base --seeds 0,1 \
  --sample-turns sampled --test-cap 40 --concurrency 8 \
  --opt-model us.anthropic.claude-opus-4-8 --out term.json

# the n=0 anchor: the same scoring with retrieval OFF (--no-rag); merged into the report as base@0.
AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_trace_scaling.py \
  terminal-tasks --counts 1 --modes base --seeds 0,1 --no-rag \
  --sample-turns sampled --test-cap 40 --concurrency 8 \
  --opt-model us.anthropic.claude-opus-4-8 --out term_norag.json

# render all three into the figure (matplotlib is ephemeral, not a project dep); base@0 is drawn
# in the "0" slot left of 10^0 with a dotted connector to the first RAG point.
uv run --with matplotlib python .agents/scripts/plot_trace_scaling.py \
  --report tau-bench=tau.json --report terminal-tasks=term.json --report swe-bench=swe.json \
  --out docs/research/trace_scaling_law --title "Trace scaling law (RAG-only)"
```

Each corpus was captured live from its real benchmark; see the capture tooling and READMEs under
`packages/environment-capture/tau-bench/`, `packages/environment-capture/terminal-tasks/`, and `packages/environment-capture/swe-bench/`.

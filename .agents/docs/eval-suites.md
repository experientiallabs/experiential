---
source: https://app.notion.com/38e0f8b3f591812b81ebd421b150e8f4
area: Evaluation
status: Current
migrated: 2026-07-02
promoted: docs/eval_suites.md (2026-07-02) — this copy is the pre-promotion snapshot; edit the docs/ version
---

# Eval suites (formerly benchmarks + leaderboard)

> **History:** this page originally described `wmh bench` + `benchmarks/<name>/benchmark.toml` + a leaderboard. That system was **removed in PR #38** ("consolidate bench into example eval suites") and replaced by the example-local eval suites described below.

An **eval suite** is a committed, reproducible eval config living next to the example it scores: `examples/<task>/evals/<suite>.toml`. It names the trace files (relative to the suite file) and pins the scoring config. `wmh eval run <suite>` scores a prompt against it and persists the result locally.

This sits on top of the open-loop eval scorer (`wmh.engine.eval`): for each held-out step it feeds the recorded `(state, action)` teacher-forced, has the world model predict the observation, and scores it against the *real* recorded observation with the reference-grounded 5-dimension `RubricJudge`.

## The definition

Each example task directory bundles everything: corpus, capture tooling, prebuilt models, and its eval suites:

```
examples/
  tau-bench/
    traces.otel.jsonl     # the committed corpus (1033 traces)
    evals/default.toml    # the suite definition (committed)
    models/               # prebuilt example world models (tau-bench, tau-telecom)
```

`evals/default.toml`:

```toml
title = "Tau Bench default replay"
description = "Open-loop reconstruction fidelity over the bundled tau-bench trace corpus."
files = ["../traces.otel.jsonl"]   # resolved relative to this file
train_split = 0.7
sample_turns = "all"
seed = 0
judge = "rubric"
```

## Running

```bash
wmh eval list                    # every suite under examples/*/evals/
wmh eval run tau-bench           # run a suite, save a local JSON result
wmh eval results                 # summarize local suite results (all suites)
wmh eval results tau-bench       # ... or one suite
wmh eval <trace files...>        # ad hoc replay scoring, no suite needed
```

Suite CLI flags (`--prompt`, `--judge`, `--train-split`, `--top-k`, …) override the suite's pinned config for one-off comparisons. Results are written under `.wmh/evals/` (local artifacts, not committed).

## How it layers

`wmh/engine/eval_suites.py` discovers suites (`examples_root/*/evals/*.toml`), resolves one by name, and lists persisted results. The `wmh eval` CLI command is a thin wrapper; scoring delegates to the same `wmh.engine.replay` path as ad hoc `wmh eval` and the research harness, so all fidelity numbers stay comparable.

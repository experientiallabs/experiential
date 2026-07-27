# Eval suites (formerly benchmarks + leaderboard)

> **History:** this page originally described `wmo bench` + `benchmarks/<name>/benchmark.toml` + a leaderboard. That system was **removed in PR #38** ("consolidate bench into example eval suites") and replaced by the example-local eval suites described below.

An **eval suite** is a committed, reproducible eval config living next to the example it scores: `examples/<task>/evals/<suite>.toml`. It names the trace files (relative to the suite file) and pins the scoring config. `wmo eval run <suite>` scores a prompt against it and persists the result locally.

This sits on top of the open-loop eval scorer (`wmo.engine.eval`): for each held-out step it feeds the recorded `(state, action)` teacher-forced, has the world model predict the observation, and scores it against the *real* recorded observation with the reference-grounded 5-dimension `RubricJudge`.

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
sample_turns = "all"
seed = 0
```

`train_split` is omitted on purpose. It defaults to the same `DEFAULT_TRAIN_SPLIT` (0.8) that
`wmo build` cuts, and both cut the SAME deterministic `trace_id` hash line, so a suite that pins a
lower value scores traces GEPA trained on and reports an inflated fidelity. Pin it only when the
model under test was genuinely built with a different ratio.

There is no judge knob: every suite is scored by the single `RubricJudge`. A pre-overhaul
`judge = "..."` line makes the suite fail validation: loading the file directly (`wmo eval
run path/to/suite.toml`) raises an error saying to delete the line, while name-based discovery
skips the suite with that same message as a warning on stderr — if a suite is missing from
`wmo eval list`, check the warnings.

## Running

```bash
wmo eval list                    # every suite under examples/*/evals/
wmo eval run tau-bench           # run a suite, save a local JSON result
wmo eval results                 # summarize local suite results (all suites)
wmo eval results tau-bench       # ... or one suite
wmo eval <trace files...>        # ad hoc replay scoring, no suite needed
```

Suite CLI flags (`--prompt`, `--train-split`, `--top-k`, …) override the suite's pinned config for one-off comparisons. Results are written under `.wmo/evals/` (local artifacts, not committed).

Both flows score on the `[models.worker]` role that `wmo providers set` writes to `.wmo/settings.toml`, falling back to bedrock/`claude-opus-4-8` when the project configured no role; `--provider`/`--model` override it for one run. Every run prints the backend it used (`scoring with openai (gpt-5.4-mini)`) and each saved result records it, because a fidelity number is only comparable against runs on the same model.

## How it layers

`wmo/engine/eval_suites.py` discovers suites (`examples_root/*/evals/*.toml`), resolves one by name, and lists persisted results. The `wmo eval` CLI command is a thin wrapper; scoring delegates to the same `wmo.engine.replay` path as ad hoc `wmo eval` and the research harness, so all fidelity numbers stay comparable.

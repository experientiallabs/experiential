# Eval suites (formerly benchmarks + leaderboard)

> **History:** this page originally described `wmo bench` + `benchmarks/<name>/benchmark.toml` + a leaderboard. That system was **removed in PR #38** ("consolidate bench into example eval suites") and replaced by the example-local eval suites described below.

An **eval suite** is a versioned, reproducible eval config that ships in the benchmark's data bundle, living next to the corpus it scores: `environment-capture-data/<task>/evals/<suite>.toml`. It names the trace files (relative to the suite file) and pins the scoring config. `wmo eval run <suite>` scores a prompt against it and persists the result locally.

This sits on top of the open-loop eval scorer (`wmo.engine.eval`): for each held-out step it feeds the recorded `(state, action)` teacher-forced, has the world model predict the observation, and scores it against the *real* recorded observation with the reference-grounded 5-dimension `RubricJudge`.

## The definition

Suites are not in this repo. They ship in the benchmark's published dataset on the Hub, alongside the corpus they score and the world model built from it, and `wmo download <benchmark>` is how you get all three. Nothing is discoverable before that download; there is no bundled copy to fall back on.

`wmo download` writes under the data root (`environment-capture-data/` in the working directory unless `$ENVCAP_DATA_ROOT` overrides it):

```bash
wmo download tau-bench
```

```
environment-capture-data/
  tau-bench/
    traces.otel.jsonl        # the corpus
    evals/default.toml       # the suite definition
    models/tau-bench/        # prebuilt world model (config.toml, card.json, metrics.json, prompts/, index/)
    models/tau-telecom/      # ... a second one, for the telecom domain
```

That layout is the same one a local `wmo build` writes, which is the point: `wmo eval list` globs `<data root>/*/evals/*.toml` and model resolution walks `<data root>/<benchmark>/models/<name>/`, so a downloaded bundle and a locally captured one are found by identical code. Suite `files` entries are relative to the suite file (`../traces.otel.jsonl`), which resolves because the corpus and the suites land in one benchmark directory.

The prebuilt models come down ready to run, so scoring a suite does not require building anything first:

```bash
wmo list --root environment-capture-data/tau-bench   # what the bundle shipped, with its metrics
wmo play --name tau-bench                            # step it; resolution spans downloaded bundles
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
wmo download tau-bench           # required first: suites arrive with the bundle, not with the repo
wmo eval list                    # every suite under environment-capture-data/*/evals/
wmo eval run tau-bench           # run a suite, save a local JSON result
wmo eval results                 # summarize local suite results (all suites)
wmo eval results tau-bench       # ... or one suite
wmo eval <trace files...>        # ad hoc replay scoring, no suite needed
```

Suite CLI flags (`--prompt`, `--train-split`, `--top-k`, …) override the suite's pinned config for one-off comparisons. Results are written under `.wmo/evals/` (local artifacts, not committed).

Both flows score on the `[models.worker]` role that `wmo providers set` writes to `.wmo/settings.toml`, falling back to bedrock/`claude-opus-4-8` when the project configured no role; `--provider`/`--model` override it for one run. A `--provider` naming a different backend than the role takes its model from *that* backend's catalog, so `--provider openai` runs OpenAI's flagship rather than asking OpenAI for a Claude id. Every run prints the backend it used (`scoring with openai (gpt-5.4-mini)`) and each saved result records it, because a fidelity number is only comparable against runs on the same model.

## How it layers

`wmo/engine/eval_suites.py` discovers suites (`examples_root/*/evals/*.toml`), resolves one by name, and lists persisted results. The `wmo eval` CLI command is a thin wrapper; scoring delegates to the same `wmo.engine.replay` path as ad hoc `wmo eval` and the research harness, so all fidelity numbers stay comparable.

# dabstep

Data-analysis QA over a shared payments dataset and a business-rules manual. Each task is a
question whose correct answer requires reading `manual.md` (it defines what "authorized", "fee",
and "fraud rate" mean — the raw columns are ambiguous on their own) and computing over the CSV/JSON
context files with real shell + pandas. The environment stages the task's context files into a
fresh workspace's `./data/` directory; the agent explores, analyzes, and submits an answer. Scoring
is deterministic (numeric tolerance 0.01, normalized string/list match, accepted alternates) — see
`environment_capture/benchmarks/dabstep.py`.

## Contents

- `data/train.jsonl` (5 tasks) / `data/test.jsonl` (5 tasks) — agent-visible tasks
  (prompt + `file_ids` + difficulty level). Train and test ids are disjoint.
- `datafiles/<file_id>` — the shared context files, committed **except** `payments.csv`
  (~23 MB, gitignored): `manual.md`, `fees.json`, `acquirer_countries.csv`,
  `merchant_category_codes.csv`, `merchant_data.json`, `payments-readme.md`.
- `gold/<task_id>.json` — gold answers (`answer` + optional `numeric` + `accept` variants), never
  staged into the agent workspace.
- `traces.otel.jsonl` — the trace corpus: **98 traces / 790 real transitions**, host-content-free
  (`environment_capture.scan_spans_jsonl` returns no findings; train split only, so the world model
  can't absorb the hidden test split's dynamics).
- `fetch_data.py` — downloads the gitignored `payments.csv` (and, with `--all`, every context file)
  from the upstream HuggingFace dataset, so a fresh clone is runnable.
- `convert_cache.py` — the converter that seeded the corpus from a frozen baseline cache of real
  runs (see provenance).
- `capture.py` — fresh real-run capture against this adapter (Bedrock agent), used to grow the
  corpus with richer multi-step trajectories.

## Running it

```bash
# 1. pull the large context file (payments.csv is gitignored)
uv run python environment-capture/dabstep/fetch_data.py

# 2. capture fresh real runs on Bedrock (each model runs the full train split)
uv run python environment-capture/dabstep/capture.py \
    --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 1 \
    --out environment-capture/dabstep/traces.otel.jsonl --append
```

## Results (2026-07-02, corpus as committed)

- **Open-loop fidelity** (suite `dabstep/default`, seed 0, Opus 4.8 target + rubric judge,
  run via `uv run wmh eval run dabstep/default --examples-root environment-capture`): mean
  fidelity **0.886**, error-flag accuracy **0.976**, n=85 held-out steps (an independent rerun
  scored 0.892 — the spread is judge noise). Structured pandas/JSON tool output reconstructs on
  par with the other structured-output corpora (bird-sql 0.864) and far above document-excerpt
  observations (financebench 0.586).

## Provenance

- **Dataset**: [adyen/DABstep](https://huggingface.co/datasets/adyen/DABstep). The committed
  `data/*.jsonl` + `gold/*.json` are the real DABstep `dev` set (the 10 gradeable questions —
  the `default` split is server-scored with no local gold), split disjointly into 5 train + 5 test.
  Task ids, `file_ids`, the train/test split, and the context files come from a prior
  materialization of the upstream dataset, reused as data; all adapter/grader code here is fresh
  (the documented thresholds are not inherited from anywhere).
- **Traces**: seeded with `convert_cache.py` from a frozen baseline cache of REAL runs over the
  same materialization (**3 traces**, model `gpt-5.4`, mean reward 0.0 — the bare baseline agent's
  heredoc quoting failed before it could submit; 2 zero-transition trajectories skipped at
  conversion), then grown with `capture.py` — **33 fresh real runs** on Bedrock
  (`us.anthropic.claude-opus-4-8` ×16 and `-4-7` ×17, several passes; mean reward 0.273, 9 solves,
  every train task covered). Converted traces keep the original run's reward; fresh captures are
  graded by this adapter's grader and carry their own model id. Multi-run trace ids never collide:
  a fresh run's task id is suffixed with its model + run tag (e.g. `dab-train-3#opus48-r1`).
  `us.anthropic.claude-opus-4-6-v1` was dropped from the capture set: it ignored the workspace
  scoping and issued host-targeting commands on every task, so every one of its trajectories was
  flagged and dropped by the hygiene audit.
- **Workspace containment**: `LocalBashEnv` refuses host-targeting commands, and the shared hygiene
  audit (`environment_capture.hygiene`) drops any trajectory that reached host filesystem content —
  data-analysis agents otherwise wander the host (`ls ~`, `find /`) looking for their data. To keep
  the corpus rich rather than thin, `capture.py` also gives the agent a workspace-scoped system
  prompt (its data is under `./data/`; host-targeting commands are blocked and invalidate the run),
  which cut the escape rate to near zero. `scan_spans_jsonl` reports no host content on the corpus.
- The recording harness echoed an ALLCAPS `*_SUBMIT` sentinel into the baseline runs' final
  command/output to mark the submission; the shared `load_baseline_cache` normalizes it to the
  neutral `SUBMIT` (apparatus protocol, not environment content — no result, path, or number is
  altered). Fresh Bedrock captures use a real `submit` tool and carry no such sentinel.

## License — read before redistributing

DABstep is published under **CC BY 4.0** (attribution). The task data, context files, gold answers,
and traces are redistributed here under CC BY 4.0 **with attribution to Adyen**
(https://huggingface.co/datasets/adyen/DABstep). Redistribution and commercial use are permitted
provided attribution is retained.

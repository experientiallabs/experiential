# dabstep

Data-analysis QA over a shared payments dataset and a business-rules manual. Each task is a
question whose correct answer requires reading `manual.md` (it defines what "authorized", "fee",
and "fraud rate" mean — the raw columns are ambiguous on their own) and computing over the CSV/JSON
context files with real shell + pandas. The environment stages the task's context files into a
fresh workspace's `./data/` directory; the agent explores, analyzes, and submits an answer. Scoring
is deterministic (numeric tolerance 0.01, normalized string/list match, accepted alternates) — see
`environment_capture/benchmarks/dabstep.py`.

## Contents

- `data/train.jsonl` (130 tasks) / `data/test.jsonl` (5 tasks) — agent-visible tasks
  (prompt + `file_ids` + difficulty level). Train and test are disjoint and no two tasks share a
  question. The 5 test + original 5 train come from DABstep's public-gold `dev` split; the other
  125 train tasks are drawn from the 450-task upstream pool with gold recovered from the leaderboard
  (see Provenance).
- `datafiles/<file_id>` — the shared context files, committed **except** `payments.csv`
  (~23 MB, gitignored): `manual.md`, `fees.json`, `acquirer_countries.csv`,
  `merchant_category_codes.csv`, `merchant_data.json`, `payments-readme.md`.
- `gold/<task_id>.json` — gold answers (`answer` + optional `numeric` + `accept` variants, plus an
  `upstream_task_id` for recovered tasks), never staged into the agent workspace.
- `traces.otel.jsonl` — the trace corpus: **687 traces / 4859 real transitions**, host-content-free
  (`environment_capture.scan_spans_jsonl` returns no findings; train split only, so the world model
  can't absorb the hidden test split's dynamics).
- `fetch_data.py` — downloads the gitignored `payments.csv` (and, with `--all`, every context file)
  from the upstream HuggingFace dataset; `--expand` appends new train tasks with gold recovered
  from the leaderboard (see Provenance), leaving the test split untouched.
- `leaderboard_gold.py` — recovers a clean gold answer per task from the dataset's published
  `task_scores` (a majority vote over officially-verified-correct submissions) and drops tasks with
  no confident answer (the answerability filter).
- `convert_cache.py` — the converter that seeded the corpus from a frozen baseline cache of real
  runs (see provenance).
- `capture.py` — fresh real-run capture against this adapter (Bedrock agent), used to grow the
  corpus with richer multi-step trajectories.

## Running it

```bash
# 1. pull the large context file (payments.csv is gitignored)
uv run python packages/environment-capture/dabstep/fetch_data.py

# 2. (optional) re-derive the expanded train split: recover gold from the leaderboard and append
#    new tasks whose question is not already committed (test split untouched). Needs the fetch extra.
uv run python packages/environment-capture/dabstep/fetch_data.py --expand

# 3. capture fresh real runs on Bedrock (each model runs the full train split)
uv run python packages/environment-capture/dabstep/capture.py \
    --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 1 \
    --out packages/environment-capture/dabstep/traces.otel.jsonl --append
```

## Results (2026-07-02, corpus as committed)

- **Open-loop fidelity** (Opus 4.8 target + rubric judge, seed 0; final 687-trace / 130-task
  corpus): mean fidelity **0.829**, error-flag accuracy **0.984**, n=1457 held-out steps.
  The earlier 0.884 (@80 resampled traces) was mildly inflated by cross-run task overlap
  (DECISIONS D33) — the task-diverse number is the honest one. Still well above
  document-excerpt observations (financebench 0.586), below bird-sql's sqlite (0.944).

## Provenance

- **Dataset**: [adyen/DABstep](https://huggingface.co/datasets/adyen/DABstep). The 5 test + first 5
  train tasks are the real DABstep `dev` set (the 10 gradeable questions — the `default` split is
  server-scored with no local gold), split disjointly. Task ids, `file_ids`, the train/test split,
  and the context files come from a prior materialization of the upstream dataset, reused as data;
  all adapter/grader code here is fresh (the documented thresholds are not inherited from anywhere).
- **Task-set expansion (recovered gold)**: DABstep's 450-task pool (`data/tasks/all.jsonl`) ships
  with empty answers, but the dataset repo also publishes `data/task_scores/*.jsonl` — per
  leaderboard submission, whether the official grader scored each `agent_answer` correct. An
  `agent_answer` on a `score == true` row is thus a ground-truth-verified correct answer straight
  from the benchmark's own scorer. `leaderboard_gold.py` aggregates these per task and takes the
  value a confident plurality of agents agree on (after dropping reasoning-trace answers), numeric
  answers carrying a `numeric` field for the tolerant match; tasks with no confident clean answer
  are dropped (the answerability filter). `fetch_data.py --expand` (seed 7, target 130) appended
  125 such tasks (`dab-train-5`…) whose question is not already committed. Every recovered gold
  self-grades to 1.0. Because the leaderboard grows over time, re-running `--expand` may recover a
  slightly different set; the committed sidecars are the frozen snapshot.
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

# Bench-VERIFY ground-truth inventory

Goal: label VERIFY trajectories by a recorded outcome, never by another LLM. Two corpora were
inspected for a clean, balanced pass/fail signal.

## bird-sql (chosen)

- Corpus: `packages/environment-capture/bird-sql/traces.otel.jsonl` (main checkout; not
  materialized in this worktree). 1993 traces / ~8300 spans, train split only.
- Recorded outcome: each trace's `wmh.trace.metadata` carries `reward` in {0.0, 1.0}, the
  deterministic EXECUTION-MATCH grade (the agent's submitted SQL and the reference SQL are each run
  against a pristine copy of the database and their result rows compared). This is a real
  deterministic grade, not an LLM label.
- Balance: 1345 pass / 648 fail across 222 distinct base tasks. Comfortably supports a 20/20
  balanced sample with distinct base tasks.
- Gold: `gold/<base_task_id>.json` holds `gold_sql` (the reference query). 242 sidecars present.
- Judge inputs reconstructed per trace:
  - instruction = the question + evidence hint (`gen_ai.prompt` -> `Step.task`).
  - answer = the submitted SQL (`metadata.final_answer`).
  - transcript = the bash exploration steps, rendered in the exact `RunResult.transcript()` format
    the production judge receives in `wmh.evals.closed_loop`.
  - gold assertion = one semantic post-condition mirroring the execution-match grade: "the final
    SQL is semantically equivalent to (returns the same result set as) <gold_sql>". The transcript
    shows the queries the agent ran and the rows they returned, so the judge has the evidence to
    decide equivalence.

Why this is a fair test of the outcome judge: the recorded reward IS execution-match against the
reference SQL, and the gold assertion hands the judge that same success condition. The benchmark
measures whether the production `GoldJudge` can reproduce the deterministic grade from the
transcript. False-passes are runs whose SQL looks plausible but returns wrong rows; false-fails are
correct-but-differently-structured queries (both appear in the sample).

## tau-bench (rejected for VERIFY)

- Corpus: `packages/environment-capture/tau-bench/traces.otel.jsonl`. 1033 traces, 818 pass / 215
  fail. `reward` in {0.0, 1.0} is present, so labels are clean.
- Problem: the recorded reward is driven by DB write-actions the transcript judge cannot see, not
  by gradeable natural-language assertions. Of 1033 traces, 986 have EMPTY `gold.nl_assertions`;
  reward_basis is `ENV_ASSERTION` (DB state) for 880, `DB+NL_ASSERTION` for 72, `DB+COMMUNICATE`
  for 47. `GoldJudge.score` returns a trivially-passed verdict when the gold list is empty, so for
  ~95% of tau traces the judge would predict PASS by construction and the accuracy figure would be
  meaningless. tau's outcome is fundamentally a state-diff grade, not a transcript-verifiable one.

Decision: run Bench-VERIFY on bird-sql only. tau-bench is the wrong shape for an outcome judge that
reads a transcript; measuring it would test the corpus, not the judge.

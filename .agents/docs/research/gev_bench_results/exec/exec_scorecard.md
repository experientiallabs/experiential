# Bench-EXEC scorecard: simulator vs real execution

The EXECUTE leg of the GEV triple. Question: when a candidate model runs a task inside the world
model instead of the real environment, do outcomes, and the RANKING between two candidate models,
agree with reality? Rank agreement is the number the routing optimizer depends on.

- Corpus / real env: `packages/environment-capture/bird-sql` (BIRD mini-dev, real SQLite
  databases, deterministic execution-match grader).
- Simulator: the committed `models/bird-sql` world model (1993 traces / 4168 steps captured from
  the TRAIN split only; serve `us.anthropic.claude-opus-4-7` on Bedrock, hashing embedder,
  top_k 5). See "Deviations" for why this is used in place of a fresh `--limit 40` build.
- Scenarios: 8 held-out TEST-split tasks, round-robin across the 4 test databases (`superhero`,
  `toxicology`, `student_club`, `california_schools`). Test tasks are new questions on databases
  the world model trained on, so the simulator must generalize, not replay.
- Candidates: haiku-4.5 and opus-4.8 (both Bedrock) as the two policy models.
- Grid: 8 scenarios x 2 candidates x k=2 passes x 2 environments = 64 episodes, 543s wall
  (~9 min), candidates run sequentially.
- Two verifiers, kept as separate columns, never blended:
  - deterministic (headline): the agent's final submitted SQL is executed against a pristine
    read-only copy of the REAL database and its rows compared to the gold query's rows. This grades
    the agent's OUTPUT, so the same grader scores a sim episode and a real episode identically,
    which is exactly what "does sim behavior succeed in reality" asks.
  - judge: an LLM `GoldJudge` (Opus 4.8) over transcript + answer. Intended as one common verifier
    across both environments; it flatlined (see below) and carries no signal on this task.

## Headline results (deterministic verifier)

| Candidate | real pass | sim pass | sim-optimism gap | outcome agreement (8 scenarios) |
|-----------|-----------|----------|------------------|----------------------------------|
| opus-4.8  | 0.812     | 0.688    | -0.125           | 0.875 (7/8)                      |
| haiku-4.5 | 0.188     | 0.125    | -0.062           | 0.75 (6/8)                       |

- Outcome agreement: the sim and real verdicts (binarized at a 0.5 pass rate per scenario) match on
  7/8 scenarios for opus and 6/8 for haiku.
- Sim-optimism gap is NEGATIVE for both candidates: the simulator is slightly PESSIMISTIC, not
  optimistic. It under-reports real success rather than over-promising, which is the safe direction
  for a router (a sim that flatters candidates would route to a model that then fails in
  production; this one does the opposite).

### Rank agreement (the router-critical number)

| Verifier | real: opus vs haiku | sim: opus vs haiku | same winner? | per-scenario agreement |
|----------|---------------------|--------------------|--------------|------------------------|
| deterministic | 0.812 vs 0.188 | 0.688 vs 0.125 | YES (opus) | 0.75 (6/8) |

The simulator picks the same winning candidate as reality (opus over haiku) both overall and by a
wide margin, and agrees with reality on the better candidate in 6 of 8 individual scenarios. The
sim preserves not just the winner but the large capability gap between the two models (real gap
0.625, sim gap 0.562). For the routing use case this is the load-bearing result: choosing a model
inside the simulator selects the same model reality would.

### Per-scenario detail (deterministic pass rate, mean of k=2)

| scenario | database | haiku real | haiku sim | opus real | opus sim |
|----------|----------|-----------|-----------|-----------|----------|
| bird-test-15 | california_schools | 0.00 | 0.00 | 0.00 | 0.00 |
| bird-test-10 | student_club | 0.00 | 0.00 | 1.00 | 1.00 |
| bird-test-0  | superhero | 1.00 | 0.50 | 1.00 | 1.00 |
| bird-test-5  | toxicology | 0.50 | 0.00 | 1.00 | 1.00 |
| bird-test-16 | california_schools | 0.00 | 0.00 | 0.50 | 0.00 |
| bird-test-11 | student_club | 0.00 | 0.00 | 1.00 | 1.00 |
| bird-test-1  | superhero | 0.00 | 0.50 | 1.00 | 1.00 |
| bird-test-6  | toxicology | 0.00 | 0.00 | 1.00 | 0.50 |

Notes: bird-test-15 requires a `RANK() OVER (...)` window column in the gold answer; every candidate
in every environment omits it and scores 0, consistently across sim and real. The single opus
sim/real disagreement is bird-test-16 (real 0.50, sim 0.00): the simulator was pessimistic, not
optimistic. The two haiku disagreements are low-signal cells flipping around the 0.5 threshold on a
candidate that mostly fails everywhere.

## Simulator lets weak candidates operate on hallucinated schemas

The execute-step defect this benchmark exists to surface. For each episode the agent's final SQL is
run against the real database and its error classified; `no_such_table` / `no_such_column` means the
agent built its query on schema the real database does not have, i.e. the world model handed it a
hallucinated schema and played along instead of returning a "no such table" error.

| Candidate | sim hallucination rate | real hallucination rate |
|-----------|------------------------|-------------------------|
| haiku-4.5 | 0.50 (8/16 episodes)   | 0.19 (3/16)             |
| opus-4.8  | 0.00 (0/16)            | 0.00 (0/16)             |

The simulator UNDER-CORRECTS the weak candidate. haiku frequently invents plausible-but-wrong names
(e.g. on california_schools it queries `scores`, `charter_number`, `school_id` when the real schema
has `satscores`, `CharterNum`, `cds`), and the world model generates plausible observations for
those fake tables rather than the error a real sqlite would raise, so the hallucination rate more
than doubles from 0.19 in reality to 0.50 in the sim. opus reads `schema.sql` first and anchors on
the true schema, giving retrieval strong anchors, so it never drifts in either environment (0/0).

Error-kind distribution (16 episodes each):
- haiku sim: 8 no_such_table, 4 wrong_result, 2 none, 1 sql_error, 1 no_sql.
- haiku real: 5 wrong_result, 4 no_sql, 3 no_such_table, 3 none, 1 sql_error.
- opus sim: 11 none, 5 wrong_result. opus real: 13 none, 3 wrong_result.

## The LLM-judge leg flatlined; deterministic grading is authoritative

All 64 episodes scored judge-fail (sim and real, both candidates). judge-vs-deterministic on the 32
real episodes: accuracy 0.50, 16 false-fails, 0 false-passes. The 0.50 is fully explained: real
deterministic pass rate is 16/32, the judge failed all 32, so it agreed only on the 16 real
deterministic-fails.

Root cause, from the transcripts (not the lead's initial "never sees rows" hypothesis, which is only
half of it): the judge often reasons CORRECTLY but is overridden by fail-closed scoring. On the
verified-correct episode bird-test-0 (deterministic 1.0, transcript shows the query returning 122),
the judge returned:

```
{"assertions": [{"assertion": "...return exactly this result set ...: (122,)",
                 "passed": true, "why": "...returns 122 as verified in transcript."}],
 "passed": true}
```

The judge said passed:true. But `GoldJudge._parse` matches the judge's echoed assertion against the
gold assertion BY EXACT TEXT, and my gold assertion embeds the expected rows after a newline
(`...ordering):\n(122,)`) while the judge echoed it back with a space (`...ordering): (122,)`). The
strings do not match after strip, so n_pass = 0 and the verdict fail-closes. This mismatch is
present on every scenario, which is why all 64 episodes read as judge-fail regardless of actual
correctness. Compounding it, the judge cannot execute SQL, so for episodes where the final result
rows are not echoed into the transcript it could not verify anyway.

Net: on this text-to-SQL task the judge column carries no usable signal, and the deterministic
execution grader is the sole trustworthy verifier (which is why it is the headline). This is exactly
the failure Bench-VERIFY targets: the outcome judge needs (a) to actually see or execute the result
set and (b) a scoring contract with a native confidence / abstain instead of brittle verbatim-echo
fail-close. Bench-VERIFY's fix is a prerequisite before the judge leg can be trusted as a
cross-environment verifier here.

## Confusion matrices (deterministic, per candidate over 8 scenarios)

opus-4.8: sim-pass/real-pass 6, sim-pass/real-fail 0, sim-fail/real-pass 1, sim-fail/real-fail 1.
haiku-4.5: sim-pass/real-pass 1, sim-pass/real-fail 1, sim-fail/real-pass 1, sim-fail/real-fail 5.
The sim-pass/real-fail cell (the mirage a search would chase) is 0 for opus and 1 for haiku,
consistent with a simulator that is pessimistic rather than optimistic.

## Cost and wall time

- Wall: 543s (~9 min) for 64 episodes, candidates sequential.
- Per episode: one agent rollout (<=12 turns) plus one Opus-4.8 judge call; the 32 sim episodes are
  additionally served by the world model's Opus-4.7 provider. Spend is dominated by the Opus-4.8
  agent-and-judge calls. Token usage was not metered in this run (the runner uses raw providers, not
  the MeteredProvider wrapper); wall time is the reliable cost proxy. No throttling or provider
  errors occurred.

## Reproduction

From the worktree root (`~/Desktop/Projects/wmh-gev-bench`), Bedrock creds in `.env` / ambient AWS:

```bash
# 1. Materialize the real BIRD mini-dev databases + splits + gold (gitignored / not committed).
uv run --with gdown python -m gdown 13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG -O /tmp/minidev.zip
unzip -q /tmp/minidev.zip -d /tmp/bird_dl
uv run python packages/environment-capture/bird-sql/fetch_data.py \
    --minidev-root /tmp/bird_dl/minidev/MINIDEV            # base: 52 train / 20 test, 4 dbs
uv run python packages/environment-capture/bird-sql/fetch_data.py \
    --minidev-root /tmp/bird_dl/minidev/MINIDEV --expand   # +170 train tasks, all 11 dbs

# 2. Smoke (1 scenario x 1 candidate x both envs x k=2).
uv run python .agents/scripts/gev_bench/exec_bench.py --smoke --k 2

# 3. Full grid (8 scenarios x 2 candidates x k=2 x both envs = 64 episodes).
uv run python .agents/scripts/gev_bench/exec_bench.py --scenarios 8 --k 2
```

Outputs: `exec/episodes.jsonl` (per-episode rows: candidate, env, task_id, db, attempt, det_score,
judge_passed, judge_fraction, error_kind, schema_hallucination, pred_sql, stop_reason, turns; this
doubles as the spot-check sheet), `exec/metrics.json`, this scorecard.

## Deviations and what was brittle

- The bird-sql `traces.otel.jsonl`, `data/`, `gold/`, and `databases/` are NOT committed on this
  branch (only the built `models/bird-sql` index is). Databases and splits were materialized fresh
  from BIRD mini-dev via `fetch_data.py` (base + expand), reproducing the documented 222-train /
  20-test corpus.
- Simulator = the committed `models/bird-sql` world model rather than a fresh `--limit 40 gev-bird`
  build. The trace corpus needed to build one is absent, and regenerating it requires a full fresh
  Bedrock capture (out of scope and far more expensive than the eval itself). The committed model IS
  the corpus-derived world model (train-split only, 1993 traces), so evaluating on held-out TEST
  scenarios is a clean generalization test and the honest smaller path.
- Scenarios were taken from the bird adapter's TEST split (`adapter.tasks("test")`), not
  `scenarios_from_traces` (which needs the absent traces). Deterministic gold exists for every test
  task, so the real-environment outcome needs no judge.
- The judge leg is unreliable on this task (see above); treat only the deterministic columns as
  load-bearing until Bench-VERIFY's judge fix lands.

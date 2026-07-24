# GEV benchmark: consolidated scorecard (2026-07-24)

One benchmark per step of the world-model triple, run empirically on real corpora with a
manual (hand-labeled or hand-attributed) leg each. Design + principles:
`.agents/docs/research/gev-benchmarks.md`. Per-bench detail: `gen/report.md`,
`verify/verify_scorecard.md`, `exec/exec_scorecard.md`.

## Headline table

| Step | Corpus | Automated result | Manual leg result |
|---|---|---|---|
| GENERATE | tau-bench, 100 traces, budget 15 | 9 scenarios, coverage 0.39, back-agreement 9/9 (tautological), solvable 4/9 | Blind labels: 7/9 precision; 1 rule-mutation, 1 ungradeable checklist |
| EXECUTE | bird-sql, 8 held-out scenarios x 2 candidates x k=2 x real+sim (64 episodes) | Rank agreement: SAME winner as reality (per-scenario 0.75); sim slightly PESSIMISTIC (gap -0.06/-0.12) | Schema-hallucination audit: sim under-corrects the weak candidate (haiku 0.50 sim vs 0.19 real) |
| VERIFY | bird-sql, 40 trajectories balanced by deterministic grade | Judge accuracy 0.80; false-pass = false-fail = 0.20; vote-agreement calibration flat (0.806 vs 0.778) | 8 disagreements attributed: 5 judge, 1 gold, 1 label, 1 ambiguous |

## The load-bearing positive

Choosing a model inside the simulator selects the model reality would (opus over haiku,
real 0.81 vs 0.19, sim 0.69 vs 0.13, capability gap preserved, sim errs pessimistic).
The routing optimizer's core assumption held on its first honest test.

## Cross-bench findings (each confirmed independently in two places)

1. **The verifier is the weakest link, and it fails closed and confident.**
   VERIFY: 6 of 8 judge errors were unanimous 3/3 votes, so agreement-based abstention
   cannot work; a native calibrated confidence field is required. EXEC: the judge leg
   flatlined (all 64 episodes failed, 16 false-fails vs deterministic). Root cause with
   transcript evidence: the judge often REASONED correctly but GoldJudge fail-closed on a
   verbatim-echo mismatch (gold assertion embedded expected rows with a newline; the
   judge echoed with a space; the by-text assertion match missed) - compounded by the
   judge being unable to execute SQL. Fixes, in order: (a) whitespace/format-tolerant
   assertion matching in GoldJudge (bug, filed); (b) feed the judge reference RESULT
   ROWS or an execution tool (prerequisite on execution corpora, not an optimization);
   (c) native calibrated confidence + abstention (WS-A6 pattern).
2. **Mint-time gates and manual labels catch DISJOINT defect classes.**
   Back-agreement passed a scenario whose task had silently mutated a policy rule (blind
   label caught it); the blind labels passed scenarios whose checklists back-agreement
   had killed at build time (6/15). Solvability flagged mostly harness artifacts (no
   user simulator on a dialogue corpus), not scenario defects. Keep all three; read each
   only for what it measures.
3. **Selection can invert the corpus.** Failure pinning + back-agreement drops turned an
   86 percent-success corpus into a 100 percent-failure eval set, with two mid-size
   clusters at zero. The eval-mirrors-traffic invariant needs a pinning cap and a
   surfaced outcome mix (filed).
4. **State grounding is the sim's real fidelity frontier.** GEN: all tau seed states are
   prose-only (capture never recorded structured state) - exactly where the simulator
   invented a wrong item mid-rollout. EXEC: the sim lets a weak candidate operate on
   hallucinated schema at 2.7x the real rate. Structured state capture (D-INGEST
   attribution fields) and schema-anchored simulation are the highest-leverage
   execute-step improvements.
5. **Ground truth is scarcer than it looks.** tau is unusable for judge benchmarking
   (986/1033 traces have empty gold; the recorded reward is a DB-state diff a transcript
   judge cannot see; GoldJudge trivially passes empty gold). Customer corpora will look
   like tau, so mined checklists + mint-time gates + the manual leg ARE the verify
   story in production; deterministic grades exist only on lucky corpora like bird-sql.

## Costs

GEN ~6 min wall (build 100s, verify 209s). EXEC 543s for 64 episodes. VERIFY ~40 judge
calls x3 votes. All Bedrock; `wmh scenarios` CLIs do not meter cost (filed as a gap).

## Reproduction

Each per-bench report carries exact commands. Worktree: `../wmh-gev-bench`, branch
`research/gev-benchmarks`. Note: bird-sql databases/splits are re-materialized by
`fetch_data.py` (gitignored by design); EXEC used the committed train-split-only
`models/bird-sql` world model, so its test-split scenarios are a clean held-out test.

# Benchmarking generate / execute / verify (GEV)

Status: designs + first empirical runs (2026-07-23, wm-create chat). Each step of the
world model triple gets its own benchmark, built empirically from real traces, with a
manual (hand-labeled) verification leg. Modeled on the judge meta-eval pattern
(`wmh/optimize/judge_quality.py`, PR #83): labeled cases pin required behavior,
controls preserve it, defect cases prove fixes.

## Principles (recorded per Silen, 2026-07-23)

1. **Realism is not likelihood.** Scenario-generation scoring must separate
   "this could not happen in this environment" (unrealistic; a real defect) from
   "this rarely happens" (unlikely; NOT a defect - rare-but-real scenarios are often
   the valuable ones). A judge that conflates them systematically prunes the tail of
   the distribution. This is a fundamental value-function / likelihood-estimation
   issue; every rubric below carries `realistic` and `likely` as SEPARATE dimensions,
   and nothing gates on `likely`.
2. **Manual verification is a standing leg, not a bootstrap.** Every benchmark keeps a
   hand-labeled sample (the meta-eval pattern), refreshed when the pipeline changes.
   Automated scores are only trusted where they agree with the labeled slice.
3. **Judges must be able to say "no signal".** The judge should emit a confidence and
   abstain below it (calibrated, WS-A6 pattern: stated confidence = calibrated
   P(correct)). Today ChecklistJudge/GoldJudge emit none - the VERIFY benchmark
   measures a vote-agreement proxy (k judge passes, temp > 0) and its calibration;
   a native confidence field is the follow-up feature, informed by that curve.
4. **Back-agreement + solvability are creation-time gates, not drift monitors.**
   They filter bad scenarios at mint time; nothing yet detects a scenario set going
   stale as traffic drifts (future work, logged in DECISIONS.md).

## Bench-GEN: scenario generation

- Corpus: tau-bench (packages/environment-capture/tau-bench), first 100 traces
  (deterministic cap), budget 15.
- Automated: corpus_coverage, cluster balance, back-agreement rate, solvability rate
  (`wmh scenarios build` + `wmh scenarios verify`).
- Manual: a blind labeling pass over every mined scenario against its provenance
  traces, 5 dimensions, 0/1 each:
  faithful (task matches what the source traces actually do), self-contained
  (runnable without unstated context), judgeable (checklist items are observable,
  correct post-conditions of the task), realistic (could occur in this environment),
  likely (typical of the corpus). Headline = precision on faithful+self-contained+
  judgeable+realistic; `likely` is reported but never gates (principle 1).

## Bench-EXEC: closed-loop execution

- Ground truth: bird-sql (packages/environment-capture/bird-sql) - real sqlite
  databases (fetch_data.py) and a deterministic grader, so "what really happens" is
  checkable without a judge.
- Design: same candidate policy, same scenarios, same verifier; run each scenario
  against (a) the real environment and (b) the world model. Report per-scenario
  outcome agreement, reward correlation, and - with two candidate models - policy
  RANK agreement (does the sim pick the same winner as reality? the number the
  routing optimizer actually depends on).
- Scale: ~8 scenarios x 2 candidates x k=2 x 2 environments, staged smoke first.

## Bench-VERIFY: the outcome judge

- Ground truth: recorded outcomes in the corpora (tau rewards; bird-sql deterministic
  grades) - no circular LLM labels.
- Design: balanced pass/fail trace sample; judge grades each source trajectory
  against its task's gold/checklist; report accuracy / false-pass / false-fail vs
  recorded outcome. Confidence proxy: k=3 judge votes at temp 0.7, vote agreement as
  confidence; calibration = accuracy per agreement bucket (3/3 vs 2/3). Error
  analysis: hand-read every disagreement and attribute it (judge wrong vs recorded
  label wrong vs assertion under-specified) - the attribution is what feeds
  /improve-judge, same as PR #83.

## Outputs

Runners in `.agents/scripts/gev_bench/`, results + reports in
`.agents/docs/research/gev_bench_results/<step>/`. Labeling sheets ship with every
run so Silen can spot-check the manual leg.

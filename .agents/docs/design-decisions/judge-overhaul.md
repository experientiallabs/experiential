# Judge overhaul: one judge, proven with a meta-eval (2026-07-02)

Branch `feat/judge-overhaul`. Two moves: collapse to a single judge, then fix its defects with
evidence.

## Collapse

`LLMJudge` (single functional-equivalence score) and the `--judge match` option are deleted;
`RubricJudge` is *the* judge, used identically as GEPA's fitness signal and the eval scorer.
Build already optimized against the rubric, so the "which judge" knob was a trap: eval suites
could silently score with a different metric than the one GEPA hill-climbed. The suite TOML
`judge` field, `--judge` flag, and telemetry `judge_mode` are gone (one way to do each thing).

## The meta-eval (wmh/optimize/judge_quality.py)

The judge is an automated component, so it gets its own eval: 12 hand-labeled cases
(controls + defect cases) with the score band a sound judge must land in, content modeled on
real tau-bench/terminal-task observations. Drivers in `.agents/scripts/`:

- `run_judge_quality.py` — grade the judge on the labeled cases (Bedrock).
- `run_judge_regression.py` — old vs new judge on identical cached real predictions.

## Defects found and fixed (baseline: 8/11 pass → after: 12/12, all verdicts valid)

1. **Unweighted mean masked factual failure.** All three baseline failures shared one signature:
   factuality ≤ 0.1 but headline ≥ 0.38, because format/realism are ~always high for any
   well-shaped emission (wrong-facts predictions scored 0.52–0.66). The judge's per-dimension
   scores were *correct* every time — the defect was aggregation. Headline is now the
   `RUBRIC_WEIGHTS` weighted mean: factuality 0.5 (it is the definition of functional
   equivalence), quality 0.2 (the judge's holistic verdict), format/consistency/realism 0.1 each
   (form diagnostics). All-equal-dimension replies score identically under both aggregations, so
   uniformly-judged steps keep their old scores. The `right-facts-wrong-shape` control (prose
   with correct facts → 0.555) guards against the headline collapsing into factuality alone.
2. **The prompt never described its input.** The payload's `empty_sentinel`, `content_length`
   fields were unexplained (the empty-prediction paragraph existed only in the deleted
   LLMJudge's prompt). The prompt now documents every payload field and pins edge rules:
   empty-vs-nonempty ≈ 0 (baseline scored it 0.38 with format=1.0), both-empty = exact match,
   flipped `is_error`/outcome = factuality failure.
3. **No bound on observation size.** Real corpora reach ~190 KB per observation
   (terminal-tasks; p99 32 KB). Content is now middle-truncated at 6000+6000 chars with an
   omitted-count marker; head+tail (not head-only) because `long-output-divergent-tail` proves a
   divergence hidden in the tail must stay visible. `content_length` always reports the full
   length.
4. **Judge failures scored as world-model failures.** A malformed reply (missing dimension,
   0–100 scale confusion, no JSON) used to become score 0.0 — indistinguishable from "prediction
   is garbage" — and silently: missing dims *defaulted to 0.0*. Now: one retry that states what
   was invalid (observed on Bedrock: a temperature-0 re-ask reproduces the same malformed reply;
   the meta-eval's divergent-tail case failed exactly this way until the retry carried feedback),
   then `JudgeResult.valid=False`. `replay`/`eval` exclude invalid steps from fidelity and report
   `n_invalid`/`total_invalid`; the meta-eval fails any case whose verdict is invalid (a 0.0
   judge crash must not vacuously pass a low band). `max_tokens` 512 → 1024.

## Fidelity comparability

The weighted headline changes what `overall_fidelity` means, so the shift was measured on a
seeded 47-step sample of real predictions across the three corpora (predictions generated once
with Opus 4.7 zero-shot and cached, then scored by both judges on Opus 4.8 —
`judge-regression.json`):

- mean fidelity old=0.701 → new=0.584 (−0.117); Spearman rank agreement **0.963**; 0 invalid.
- steps with new-judge factuality ≥ 0.9 (n=14): shift **+0.006** — well-predicted steps are
  unmoved.
- steps with factuality ≤ 0.3 (n=24): shift **−0.202** — the drop is concentrated exactly on
  wrong-fact predictions, which is the correction the meta-eval demanded (every one of the 8
  largest per-step drops has factuality ≤ 0.4).

So the headline drop is the defect being fixed, not noise: rankings are preserved and correct
predictions score as before. Published research numbers (e.g. the trace-scaling figure) were
produced by the judge at their commits and remain reproducible there; new runs use the
overhauled judge and are not directly comparable in absolute level.

## Raw results

`.agents/docs/research/raw/judge-quality-baseline.json`, `judge-quality-fixed.json`,
`judge-regression-preds.json` (prediction cache), `judge-regression.json`.

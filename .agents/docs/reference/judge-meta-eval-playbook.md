# Playbook: hand-labeled meta-evals for improving automated components (the judge method)

How the judge overhaul (PR #83) was actually done, end to end, so the next agent improving ANY
automated component — a judge, a retriever, a reward model, an optimizer prompt — can rerun the
method instead of rediscovering it. The concrete artifacts referenced live in
`wmh/optimize/judge_quality.py` (the suite), `.agents/scripts/run_judge_quality.py` /
`run_judge_regression.py` / `plot_judge_overhaul.py` (drivers), and
`.agents/docs/research/raw/judge-*.json` (every run). Finished narrative:
`.agents/docs/research/judge-overhaul-writeup.md`.

## The loop in one line

Ground in real data → hand-label a case suite → **baseline before touching anything** → one
failing case per defect → fix → controls must not move → stability rerun → regression on frozen
predictions → (optionally) sweep models for the best operator of the component.

## 1. Ground in real data before hypothesizing (AGENTS rule 12)

Don't invent defects from the armchair — measure the input distribution first. What found our
defects:

```bash
# Observation-length distribution across corpora (found: 190KB max, p99 32KB → truncation defect;
# 114/685 + 273/1868 EMPTY observations → the both-empty case is common, not an edge case)
uv run python -c "
from wmh.ingest import get_adapter
a = get_adapter('otel-genai')
for f in ['examples/tau-bench/traces.otel.jsonl','examples/terminal-tasks/traces.otel.jsonl']:
    lens = sorted(len(s.observation.content) for t in a.from_file(f) for s in t.steps)
    n=len(lens); print(f, 'empty:', lens.count(0), 'p50:', lens[n//2], 'p99:', lens[int(n*.99)], 'max:', lens[-1])"
```

Also diff the component against its own history: the rubric prompt had silently LOST the
empty-prediction guidance that existed only in the deleted LLMJudge prompt — found by reading
both prompts side by side, not by testing.

## 2. Build the hand-labeled suite

Shape (see `JudgeCase`): `(id, defect-tag, rationale, inputs, expected ScoreBand, optional
per-dimension bands)`. The decisions that mattered:

- **Controls before defect cases.** Five controls pin correct behavior (exact match, cosmetic
  reordering, volatile values, wrong computed values, fabricated data) so a fix that
  overcorrects is caught immediately. When a fix could overshoot, add a **counter-control**:
  the weighted headline could have collapsed into factuality-only, so
  `right-facts-wrong-shape` (correct facts as prose → must NOT score high) guards the other
  direction.
- **Bands, not point targets, and generous ones.** `[0.85, 1.0]` for "perfect", `[0, 0.35]`
  for "hard failure". You are pinning direction and magnitude class, not decimals — tight
  bands make the meta-eval flaky and push you toward tuning to one model.
- **Content modeled on real corpus steps** (tau-bench reservation JSON, terminal stdout, real
  bash actions), trimmed to a few hundred chars. Long-content cases are generated
  programmatically and sized to actually EXERCISE the truncation path (case > head+tail limit),
  with the divergence placed in the tail — that's what proves head-only truncation would lie.
- **Structural invariants get unit tests, not eyeballs**: the reordered-JSON control asserts
  `json.loads(a) == json.loads(b)` in `judge_quality_test.py`, so editing one twin can't
  silently turn the control into a wrong-facts case.
- **An invalid verdict fails its case regardless of score.** We shipped this only after being
  burned: a judge crash scored 0.0 and vacuously "passed" a low band (the divergent-tail case).
  Never let infrastructure failure impersonate a graded verdict.

## 3. Baseline BEFORE fixing, and read every output

Run the suite against the untouched component and read all verdicts (`-v` prints critiques):

```bash
uv run python .agents/scripts/run_judge_quality.py -v --out baseline.json
```

Look for the *signature across failures*, not individual failures. Ours: all three failures had
factuality ≤ 0.1 but headline ≥ 0.38, while the per-dimension scores were CORRECT every time →
the defect was the aggregation (unweighted mean), not the judging. That one observation turned
"rewrite the prompt" into "reweight the mean + small prompt additions", which is why controls
didn't move.

## 4. Fix loop discipline

- One failing case (live) or failing unit test (deterministic) per defect, watched failing
  first. Parsing/validity/truncation are deterministic → unit tests; scoring judgment is
  model-behavior → meta-eval cases.
- Choose fixes that provably don't move the rest: the weighted mean was designed so any
  all-equal-dimensions reply scores identically to the old mean → uniformly-judged steps are
  untouched by construction (pinned in `judge_test.py`).
- Rerun the FULL suite after each fix; controls moving = overcorrection.
- Rerun once more at the end for stability (ours reproduced within ±0.09/case). One green run
  can be luck; two is a result. A case that only passes via the retry path is worth noting.

## 5. Regression on frozen predictions (the comparability proof)

To show a metric change is a targeted correction, hold the inputs constant: generate
world-model predictions ONCE, cache them, then score the same cache with old and new component
(`run_judge_regression.py` — the old judge is snapshotted verbatim from git into the script):

```bash
uv run python .agents/scripts/run_judge_regression.py \
  --cache .wmh/judge-regression-preds.json \
  --out   .wmh/judge-regression.json
```

Report three things: **rank agreement** (Spearman; ours 0.963), **shift sliced by the
load-bearing dimension** (factuality ≥0.9: +0.01; ≤0.3: −0.20 — the shift must concentrate
where the defect was), and **the biggest per-step deltas read as texts** — numbers alone can
hide a miscalibrated subgroup (this reading is also where we accepted the one deliberate
strictness: empty prediction vs warning-only output → 0.0, correct GEPA incentive).

## 6. Sweeping for the optimal operator (which model should judge?)

The same suite doubles as a benchmark of candidate models — pass rate saturates, so rank by
calibration: control mean (→1.0), hard-defect mean (→0.0), and their **separation**:

```bash
uv run python .agents/scripts/run_judge_quality.py --model us.anthropic.claude-opus-4-7 --out m.json
uv run python .agents/scripts/run_judge_quality.py --model gpt-5.5 --provider openai --out g.json
```

2026-07-02 result: all five candidates 12/12 (the rubric is model-robust, not Opus-tuned);
Opus 4.8 best separation (0.907), Sonnet 4.6 the budget alternative. Switching judge models =
re-baselining all fidelity numbers. Ops gotchas hit during the sweep: endflow account gets
AccessDenied invoking Opus 4.8/4.7 (ListInferenceProfiles listing ≠ invocable); default-account
4.8 brownouts under load — which is why the judge is PINNED (never rides `.wmh/fallback.toml`;
see `docs/reference/failover.md` once PR #51 lands) while world-model calls fail over.

## 7. Judge-runtime facts worth not relearning

- **Retry-with-feedback, not blind retry.** At temperature 0 an identical re-ask reproduces the
  same malformed reply byte-for-byte (observed on Bedrock); the retry must state what was
  invalid. Malformed = missing dimension, out-of-range value (0–100 scale confusion — do NOT
  clamp 85 to 1.0), or no JSON. After the retry: `valid=False`, excluded from aggregates
  everywhere (replay/eval exclude+count; GEPA imputes batch mean and drops the critique from
  reflection; `score_prompt` raises on total outage).
- **Cost/latency**: one 12-case meta-eval run on Opus 4.8 ≈ a dollar-ish and ~1 min at
  concurrency 3–4; the 47-step regression (predictions + 2×judging) is the expensive part —
  which is exactly why predictions are cached and reused.
- **Figure regeneration**: `uv run python .agents/scripts/plot_judge_overhaul.py` re-renders
  `.agents/docs/research/judge_overhaul.png` from the raw JSONs (brand palette per AGENTS rule 15).

## 8. Extending the suite

Add cases when: a new corpus type arrives (e.g. financebench numeric observations — likely
wants NumericJudge-boundary cases), a judge disagreement is found while reading real eval
scorecards (turn the disagreement into a case with a band and a rationale), or a new failure
mode ships (every prompt/parsing change gets its guarding case in the same PR). Keep ids
stable — the raw run JSONs are keyed by case id and lose comparability if ids churn.

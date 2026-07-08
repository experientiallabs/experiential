---
name: improve-judge
description: Iteratively improve the RubricJudge (or any LLM scorer) against a hand-labeled dataset - run the judge, identify false positives/negatives, diagnose why each failed, propose one experiment (prompt, model, or context) per failure class, and prove the fix without regressing controls. Use when the user distrusts judge scores, asks to improve/calibrate/tune the judge, or a new corpus type needs judge coverage.
---

# Improve the Judge

An iterative calibration loop against a hand-labeled dataset. Never tweak the judge from
intuition: every change starts from a disagreement you can point at and ends with a case that
would catch its regression. Deep background and worked example, if it still exists (`.agents/`
is prunable): `.agents/docs/reference/judge-meta-eval-playbook.md`.

The loop (repeat until a full pass produces no new disagreements you'd act on):

## 1 — Run the judge on the dataset

The dataset is `JUDGE_QUALITY_CASES` in `wmh/optimize/judge_quality.py`: hand-labeled
(action, actual, predicted) triples with the score band a sound judge must land in. Run it
against the pinned judge model (never a failover chain — chains make scores incomparable).
The stable entry point is the Python API (write run outputs somewhere UNCOMMITTED — `.wmh/` or
`/tmp` — they are working data, not repo content):

```bash
uv run python - <<'PY'
import json
from wmh.optimize.judge import RubricJudge
from wmh.optimize.judge_quality import run_judge_quality
from wmh.providers import ProviderConfig, ProviderKind, get_provider

judge = RubricJudge(get_provider(ProviderConfig(
    kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8")))
report = run_judge_quality(judge, concurrency=4)
for v in report.verdicts:
    print("PASS" if v.passed else "FAIL", v.case_id, f"{v.score:.3f}", v.failures or v.critique[:90])
print(report.summary())
open("/tmp/judge-iter.json", "w").write(report.model_dump_json(indent=2))
PY
```

(`.agents/scripts/run_judge_quality.py` is a convenience driver for the same thing — use it if
it's still around, but don't depend on it.)

If the concern came from real eval runs, ALSO pull disagreements from the wild: read per-step
scorecards from a recent `wmh eval` result (`.wmh/evals/**.json` carries `predicted`, `actual`,
`score`, `dimensions`, `critique` per step), sample ~20 steps across the score range, and ask of
each: "do I agree with this verdict?" Every disagreement becomes a labeled case (see step 5 of
the playbook for band-writing rules). The dataset must grow from real data, not toy strings.

## 2 — Identify the false positives and negatives

Classify every miss against the human label — the direction tells you what kind of defect
you have:

- **False positive** (judge scored ABOVE the band): the judge was fooled — well-shaped but
  factually wrong output, fabricated data, flipped outcome scored as plausible.
- **False negative** (judge scored BELOW the band): the judge was over-harsh — volatile values
  punished as wrong facts, cosmetic differences punished as format errors, legitimate empty
  output punished as omission.
- **Invalid verdicts** (`valid=False` after the retry): infrastructure, not judgment — count
  them separately; an invalid verdict must never pass a case vacuously.

Also recheck the controls: a previous fix that overcorrected shows up as a NEW false negative
on a control, not as a defect-case failure.

## 3 — Identify why each failed

Read the critique and per-dimension scores of every miss before hypothesizing — the judge
usually tells you. Attribute each miss to exactly one layer:

- **Aggregation** — dimensions were scored correctly but the headline hides it (the original
  overhaul's defect: factuality ≤ 0.1 masked by format/realism ≈ 1.0 under an unweighted mean).
- **Prompt rule gap** — a scenario the system prompt never defines (empty-vs-nonempty,
  both-empty, flipped `is_error`, deterministic-vs-volatile content). The critique reveals the
  judge inventing its own policy.
- **Context** — the judge couldn't SEE what it needed: payload fields it wasn't told about,
  truncation hiding the divergent region, missing action arguments, oversized input.
- **Model capability** — the rubric is fine but the model applies it loosely (weak
  separation between controls and hard defects, unstable scores across reruns).
- **Parse/protocol** — malformed replies, missing dimensions, scale confusion (85 on a 0-100
  scale must invalidate, never clamp to 1.0).

## 4 — Propose ONE experiment per failure class, expectation first

Before changing anything, encode the expectation: add or tighten a case in
`JUDGE_QUALITY_CASES` (or a unit test in `judge_test.py` for deterministic layers) and watch it
fail. Then run exactly one experiment matched to the layer:

- **Prompt** — add/sharpen the edge rule or reweight `RUBRIC_WEIGHTS`. Prefer changes that are
  provably inert elsewhere (the weighted mean was chosen so all-equal-dimension replies score
  identically to before). Guard against overcorrection with a counter-control (e.g.
  `right-facts-wrong-shape` stops the headline collapsing into factuality-only).
- **Model** — sweep candidates with the same suite (the snippet above with a different
  `ProviderConfig`) and rank by CALIBRATION, not pass rate: control mean → 1.0, hard-defect
  mean → 0.0, and their separation. Switching judge models re-baselines every fidelity number —
  say so explicitly, and bump `JUDGE_VERSION` in `wmh/optimize/judge.py` for any change to
  scoring semantics so persisted results stay distinguishable.
- **Context** — change what the judge sees: payload fields (documented IN the prompt),
  truncation head/tail limits, retry feedback wording, `max_tokens`.

## 5 — Prove it, then re-anchor

- Full suite green, **controls unmoved** — a fix that shifts controls is an overcorrection.
- Rerun once more for stability (one green run can be luck; two is a result).
- If scoring semantics changed (weights, prompt rules): rerun a frozen-prediction regression —
  generate world-model predictions ONCE on a seeded step sample, cache them to a file, score the
  same cache with the old judge (snapshot its prompt/aggregation from git) and the new one, and
  report Spearman + shift sliced by factuality band; the shift must concentrate where the defect
  was. (A ready-made driver may exist at `.agents/scripts/run_judge_regression.py`.)
- Commit the new cases with the fix and bump `JUDGE_VERSION` if semantics changed. Run outputs
  are working data: keep them out of git (`.wmh/`, `/tmp`); commit only the few small, stable
  result JSONs a finished writeup actually cites.

Stop when step 1 + a fresh scorecard sample produce no disagreement worth a case. Do not stop
on a green suite alone — the suite only contains yesterday's disagreements.

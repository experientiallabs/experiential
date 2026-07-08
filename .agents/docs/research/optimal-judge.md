# Optimal judge model (2026-07-02)

Question: with the judge pinned off the failover chain (PR #51), which model should judge?
Method: the judge-quality meta-eval (12 hand-labeled cases, `wmh/optimize/judge_quality.py`)
run per candidate, each pinned; compared on pass rate and calibration. Raw runs in `raw/`
(`judge-model-*.json`, plus the two `judge-quality-fixed*.json` Opus 4.8 runs from PR #83).

| judge model | account | pass | high-band controls (→1.0) | hard-defect mean (→0.0) | separation |
|---|---|---|---|---|---|
| **Opus 4.8** | default only¹ | 12/12 ×2 | **1.000** | **0.093–0.095** | **0.907** |
| Opus 4.7 | default | 12/12 | 0.988 | 0.105 | 0.883 |
| Sonnet 4.6 | endflow | 12/12 | 0.988 | 0.109 | 0.879 |
| GPT-5.5 | OpenAI | 12/12 | 1.000 | 0.126 | 0.874 |
| Opus 4.6-v1 | endflow | 12/12 | 0.970 | 0.130 | 0.840 |

¹ endflow cannot invoke Opus 4.8/4.7 ("not available for this account" — inference profiles
are listed but access is not granted). Discovered during the sweep.

## Conclusion

**Opus 4.8 on the default account stays the pinned judge** — best on every axis, and stable
across two runs (±0.002). This composes with the failover ladder: world-model prediction load
drains through the endflow rungs first, which keeps default-account capacity free for exactly
the calls that must not fail over (the judge).

- Budget alternative: **Sonnet 4.6** (separation 0.879 at a fraction of the cost) if judge
  spend ever dominates; scores are NOT directly comparable across judge models, so switching
  means re-baselining.
- GPT-5.5 is a viable cross-vendor sanity check (perfect on controls) but is the loosest of
  the top four on hard defects.
- All five models pass the meta-eval outright, which is itself evidence that the overhauled
  prompt + weighted aggregation (PR #83) are robust to the judging model rather than tuned to
  Opus 4.8.

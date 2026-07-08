---
source: https://app.notion.com/38e0f8b3f59181669a23e295138efbec
area: Evaluation
status: Current
migrated: 2026-07-02
---

# Benchmark results: reproducibility

> **Dated snapshot (June 2026).** These numbers were measured on the then-committed 66-trace tau2
> corpus; the corpus has since grown to 1033 traces and the repo layout changed (PR #38: paths are
> now `examples/tau-bench/...`, and the committed per-step report JSONs under `benchmarks/results/`
> were removed). The methodology and the headline conclusion (the GEPA lift is stable, not
> run-to-run luck) stand; rerun the commands below for current numbers.

The base-vs-GEPA comparison comes from open-loop reconstruction fidelity (`wmh eval`) on the
committed tau-bench corpus (at the time: 66 traces / 433 steps; telecom + airline + retail, captured
from Sierra's real tau²-bench). This doc records the exact methodology so the numbers can be
regenerated.

## Reproduce

Requires Bedrock credentials (Opus 4.8 is the only live backend here). The runs cost roughly
$1–2 each and take a few minutes (84 held-out steps × judge calls).

```bash
# Base prompt (the un-evolved BASE_ENV_PROMPT)
AWS_REGION=us-east-1 uv run wmh eval examples/tau-bench/traces.otel.jsonl \
  --region us-east-1 --judge rubric --train-split 0.7 --seed 0 \
  --out base_report.json

# GEPA-optimized prompt (the committed example model)
AWS_REGION=us-east-1 uv run wmh eval examples/tau-bench/traces.otel.jsonl \
  --region us-east-1 --prompt examples/tau-bench/models/tau-telecom/prompts/optimized.txt \
  --judge rubric --train-split 0.7 --seed 0 \
  --out optimized_report.json
```

`--train-split 0.7 --seed 0` deterministically selects the same 11-trace / 84-step held-out split
both times, so the two runs are comparable. Each `*_report.json` carries per-step scores, per-step
rubric dimensions, and the judge critiques.

## Results obtained (2026-06, Bedrock Opus 4.8, top-k=5 retrieval)

(The per-step report JSONs referenced here were committed under `benchmarks/results/` at the time
and removed in PR #38; regenerate with the commands above.)

| Prompt | held-out steps | fidelity (mean ± std) | error-flag acc |
| --- | --- | --- | --- |
| Base | 84 | ~0.74 ± 0.35 | ~0.80 |
| GEPA-optimized | 84 | ~0.86 ± 0.20 | ~1.00 |

Per-dimension (rubric judge), optimized prompt: format ~0.99, factuality ~0.72, consistency ~0.88,
realism ~0.97, quality ~0.76.

**On variance / repeatability (multi-run hardening).** The LLM judge is non-deterministic, so the
same split scores slightly differently run to run. Repeating both evals on the identical 84-step
holdout:

| Prompt | run 1 | run 2 | mean ± std |
| --- | --- | --- | --- |
| Base | 0.755 | 0.723 | 0.739 ± 0.016 |
| GEPA-optimized | 0.864 | 0.854 | 0.859 ± 0.005 |

The two distributions **do not overlap** (worst optimized 0.854 > best base 0.755), so the
**+0.12 lift is stable, not run-to-run luck**. Treat the headline table as approximate (≈±0.02
cross-run on top of the per-step std). The committed report JSONs are run 2 (base 0.723, optimized
0.854). Both runs use the same single seed (`--seed 0`), so this measures judge non-determinism on
one split — not seed-to-seed variance, which remains a GEPA-research follow-up.

## Caveats

- **One corpus, 84-step holdout** (per-step std ±0.19–0.34; cross-run ≈±0.02). Directional, not a
  leaderboard. More benchmarks/larger holdouts would tighten it further.
- The judge is an LLM (Opus 4.8) at temperature 0, but still has some variance; the per-step scores
  in the committed reports are a single sample each.
- Retrieval uses the offline lexical `HashingEmbedder` (semantic phi untested).
- `held_out_accuracy` in `examples/tau-bench/models/tau-telecom/metrics.json` (0.675) is GEPA's *internal*
  validation score over its own 317-rollout search — a different measurement from these `wmh eval`
  fidelity numbers; don't conflate them.

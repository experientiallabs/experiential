# WM scenario mining × filtered-BC ablation — mined vs random selection

**Status:** complete · **Dates:** 2026-07-04 → 07-06 · **Repo:** world-model-harness PR #81 (`feature/scenario-set-construction` @ 3b64cae) · **Driver:** Claude (Fable)

## Question
Do scenarios *mined* from traces (facets → cluster → hybrid select) make better **training data** than scenarios picked uniformly at random — everything else held equal?

## Setup (identical across all arms)
Base student Qwen3.5-9B self-rolls each scenario 6× in a frozen gpt-5.4 world model (Azure Foundry); Opus 4.8 (Bedrock us-east-2) checklist-judges each episode; ≤2 passing episodes kept per scenario ("filtered BC" — no teacher, no hint). LoRA r=32 α=64 lr=1e-4, 3 epochs, completion-only loss, think blocks preserved. Eval: 21 held-out mined scenarios × **3 passes (all numbers k=3 means)**, same WM + judge. Pools: 60 scenarios/arm, same gpt-5.4 synthesis + Opus back-agreement validation — only SELECTION differs.

## Results

| arm | success | pass-rate | notes |
|---|---|---|---|
| base | 27.0% | 0.425 | |
| bc-mined v1 (summary-only embeddings) | 22.2% | 0.488 | retail collapsed to 0% |
| bc-mined v2 (domain+tools embeddings) | 30.2% | 0.537 | retail fixed: 33.3%, best of all arms |
| bc-random seed 0 | 36.5% | 0.581 | the lucky draw |
| bc-random seed 1 | 27.0% | 0.589 | |
| bc-random seed 2 | 27.0% | 0.524 | |
| **random mean ± sd (3 seeds)** | **30.2% ± 5.5** | **0.564 ± .035** | |

Paired per-scenario, v2 vs mean-of-3-random: success 6W/6L/9T, Δ +0.000; pass-rate 11W/9L/1T, Δ −0.027.

## Verdict
1. **Summary-only mined selection is genuinely worse than random** (−8.0 pts vs the 3-seed mean): phrasing-level clustering split capabilities into duplicate clusters and drifted the domain mix away from the corpus (47% telecom in kept episodes vs 87% for random; 15% airline — absent from eval).
2. **Capability-enriched embeddings (`[domain] summary | tools: sig`, wmh 454d93f) recover to exact parity** with the random mean on success (30.2% = 30.2%) and fix the retail collapse outright; still −0.027 pass-rate. Mining does **not** beat random on this corpus.
3. **Method-level finding:** filtered self-BC itself is worth +3.2 pts success / +0.14 pass-rate over base on average, with **±5.5-pt draw-to-draw noise** — single-arm comparisons on this eval cannot support improvement claims; use multi-seed means.
4. Residual mined-arm gap is telecom-only → next single-variable lever: corpus-proportional domain quotas in the 70/30 allocator (untested).

## Caveats
Eval pool telecom-heavy (18/21), no airline; all numbers inside the WM (no real-env leg for BC arms); n=21 scenarios/arm.

## Artifacts
`world-model-harness-scenario-construction/.agents/docs/research/distill/` — pools, kept episodes, per-arm eval JSONs, PR comment draft; journal: `claas-verl/experiments/tau/07_03_2026_wm-mined-scenario-distill.md`; clarity report: `ablation_clarity_report.html`.

## Filtered-BC ablation: mined vs random scenario selection — result

The decisive experiment for whether mined scenarios beat randomly-selected scenarios as a *training* signal. Clean ablation: same synthesis + validation for both pools (only SELECTION differs), same self-BC collection (base Qwen3.5-9B, 6 samples/scenario, keep ≤2 judge-passing episodes), identical LoRA hyperparams, same eval (21 held-out scenarios × 3 passes, gpt-5.4 WM, Opus 4.8 judge).

| arm | success | pass-rate | retail | telecom | train data |
|---|---|---|---|---|---|
| base | 27.0% | 0.425 | 22.2% / 0.33 | 27.8% / 0.44 | — |
| bc-mined | 22.2% | 0.488 | 0.0% / 0.46 | 25.9% / 0.49 | 73 eps / 321 samples |
| bc-random | **36.5%** | **0.581** | 22.2% / 0.54 | 38.9% / 0.59 | 48 eps / 206 samples |

Paired per-scenario (n=21, two-sided sign tests): mined vs random success **3W/8L/10T** (Δ −14.3 pts, p=0.23), pass-rate **7W/11L/3T** (Δ −0.093, p=0.48). Random vs base: +9.5 pts success, +0.156 pass-rate (p=0.39 / p=0.10).

**Verdict: the mining-helps-training hypothesis is not supported.** No comparison reaches significance at n=21, and every point estimate favors random — despite the mined arm having *more* training data (mined scenarios are more collectible under self-BC: 73 vs 48 kept episodes) and the eval pool itself being mined, which if anything should bias toward the mined arm.

What this does and doesn't say:
- It does NOT undercut the mining pipeline's other role — validated eval-set construction — nor the earlier end-to-end result (which used a privileged teacher, a different treatment).
- It DOES say cluster-based mined selection is not a better *training-data selector* than uniform random in this self-BC setup. Post-hoc analysis points at two data-backed mechanisms: (1) **distribution shift** — diversity-seeking selection flattened the domain mix (mined kept-episodes: 47% telecom, 15% airline-which-the-eval-lacks) while random matched the telecom-heavy corpus/eval (87% telecom); (2) **within-domain concentration** — the gap persists within telecom alone (paired 3W/7L/8T, −13 pts) where both arms had comparable data, and 13 of 35 mined telecom scenarios are MMS-troubleshooting variants. Cluster-diverse ≠ instance-diverse. A corpus-proportional-quota variant of mining would isolate (1) from (2).
- Statistical honesty: n=21 scenarios → wide CIs; the claim is "no advantage, direction reversed", not "random is significantly better".

**Update (2026-07-06b): random-seed variance test.** The single random arm above turned out to be a lucky draw. Two more full random arms (independent seeds, identical pipeline) landed at 27.0% success each — equal to base. Random over 3 seeds: **30.2% ± 5.5 success, 0.564 ± .035 pass-rate**. bc-mined-v2 (below) sits at **exactly the random mean** on success (30.2%; paired vs per-scenario mean-of-3-seeds: 6W/6L/9T, Δ +0.000) and −0.027 behind on pass-rate. Sharpened verdict: summary-only mined selection is genuinely worse than random (−8 pts vs the mean); capability-enriched mined selection is at parity; nothing beats random yet; and self-BC itself is worth about +3 pts success / +0.14 pass-rate on average, with ±5.5-pt draw-to-draw noise that any future improvement claim must clear.

**Update (2026-07-06a): enriched-embedding rematch.** We re-ran the mined arm with one fix — facet embeddings now carry domain + tool signature instead of the bare task summary (`454d93f`), so clustering groups by capability rather than phrasing. Result: **bc-mined-v2 = 30.2% success / 0.537 pass-rate** — recovers most of the deficit (paired vs random now 3W/5L/13T, −6.3 pts, vs 3W/8L/10T, −14.3 pts before) and outright fixes the retail collapse (0% → 33.3%, best of all four arms). Mined selection is now statistically indistinguishable from random, but still doesn't beat it; the residual gap is entirely telecom, i.e. the still-unfixed 70/30 allocator drift. Next single-variable test: corpus-proportional domain quotas in selection.

Artifacts in `.agents/docs/research/distill/` @ fd6ed5b (pools incl. v2, kept episodes, per-arm eval JSONs). Full protocol + audit trail in the experiment journal.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

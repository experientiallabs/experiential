# Scenario-set construction: e2e verification on tau-bench (small-model run)

Date: 2026-07-02 · Branch: `feature/scenario-set-construction` · Raw results:
[`scenario_e2e_results_tau_bench.json`](./scenario_e2e_results_tau_bench.json) · Runner:
[`.agents/scripts/run_scenario_e2e.py`](../../scripts/run_scenario_e2e.py) (+
`run_scenario_test1_domain.py` for the corrected Test 1)

## Setup

- **Corpus**: 60 tau2-bench traces (20 per domain: airline, retail, telecom) from
  `examples/tau-bench/traces.otel.jsonl`.
- **Models** (Anthropic-on-Bedrock is gated on this account, so everything runs on small AWS
  models): Nova Lite for facets, cluster naming, synthesis, checklist judge, and the world model
  serve provider (prebuilt `examples/tau-bench/models/tau-bench` artifact); Nova Micro + Nova
  Lite (t=0.0 / t=0.9) as the 4 agent configs; Titan v2 (512d) for facet embeddings.
- **Pipeline under test**: facets → Titan embed → k-means → SemDeDup → hybrid-allocation medoid
  selection (K=8 out of a 30-scenario pool) → WildBench-style synthesis → closed-loop verification.
- Every reported metric is the mean of 3 passes (house rule); wall clock 477s + ~90s follow-up.

## Results

### Test 1 — clustering ground-truth recovery (corrected to domain labels)

This corpus records exactly **one rollout per task_id** (1033 traces, 1033 labels), so
task-identity recovery is degenerate (k = n scores perfectly for any clustering). The available
non-trivial ground truth is the tau2 **domain**, k=3:

| embedding | purity | ARI |
|---|---|---|
| facet summaries (ours) | 0.667 | 0.399 |
| raw trace digests (baseline) | **1.000** | **1.000** |

Reading: the baseline wins **by construction** — raw digests contain domain-specific tool names
and schemas, so domain separation is trivial. Facet summaries abstract to task intent, which
legitimately crosses domains ("cancel a booking" exists in airline and retail). The two
embeddings capture different axes (surface/domain vs intent); intent is the axis a representative
eval set needs to cover, and domain is recoverable from metadata anyway. **The real
task-identity test requires a corpus with multiple rollouts per task** — follow-up: capture one
(cf. `capture_telecom_multimodel.py`) or use TRAIL failure categories as labels.

### Test 2 — predictive fidelity (does K=8 predict the 30-scenario pool?)

Score matrix: 4 agent configs × 30 pool scenarios × 3 passes (360 world-model episodes, max 5
steps). Actual full-pool scores: micro-t0 0.760, micro-t0.9 0.776, lite-t0 0.793, lite-t0.9 0.817.

| method | MAE | Spearman | Kendall |
|---|---|---|---|
| **ours-k8 (weighted)** | **0.035** | −0.40 | −0.33 |
| random-k8 seed0 | 0.009 | +1.00 | +1.00 |
| random-k8 seed1 | 0.057 | +0.74 | +0.55 |
| random-k8 seed2 | 0.079 | +1.00 | +1.00 |

Reading: ours beats the random-K **average** on MAE (0.035 vs 0.048) and 2 of 3 seeds. The rank
correlations are **underpowered here, for ours and for the baselines alike**: the four agents'
true scores span only 0.057 — within rollout noise — so orderings over 4 near-ties are noise
(note seed0/seed2 hitting a "perfect" +1.00, which is luck, not signal). The harness reports the
right metrics; this particular agent lineup is too closely matched to rank. Follow-up: rerun with
a real capability spread (e.g. Nova Micro vs Claude/GPT-class) and more configs.

### Test 4 — closed-loop verification of the K=8 set

Nova Lite agent rolled against the tau-bench world model, graded by the generated checklists:

- **Solvable**: 5/8 (62%) — the 3 failures have partial pass rates (0.33–0.67), i.e. scenarios
  where the baseline agent completes some but not all checklist criteria.
- **Back-agreement**: 5/8 (62%) — the checklist judge, grading each scenario's own source
  trajectory, matches the recorded tau2 reward 5 times of 8.

Reading: this is exactly the filter working as intended — `wmh scenarios verify --drop` would
keep the 4/8 scenarios passing both checks. With Nova Lite as both judge and agent these rates
are a floor, not a ceiling; the disagreement cases are worth reading individually (rule 12) to
split judge error from checklist error.

## Honest limitations

1. Single corpus (tau-bench), single seed for clustering/selection, small pool (30) and budget (8).
2. Nova Lite is judge, agent, synthesizer, AND world-model server — correlated failure modes;
   the Test-2 "ground truth" itself lives inside the world model (fidelity of the WM bounds this).
3. Task-identity recovery untested (corpus limitation); failure pinning untested (facet extractor
   labeled ~all subsampled episodes success/unknown — no failure categories to pin).
4. Rank-correlation metrics need a wider agent capability spread to be meaningful.

## Correction (2026-07-03): selection vs random, properly powered

The 3-seed random baseline above is misleading. Recomputed offline on the saved score matrix
(`.agents/scripts/analyze_selection_vs_random.py`, results in
`selection_vs_random_correction.json`):

| selection | MAE | percentile vs 2000 random draws | clusters covered |
|---|---|---|---|
| uniform random K=8 (2000 draws) | 0.045 mean / 0.040 median | — | ~5.2/8 mean |
| ours as run (**script bug**: pool re-clustered at k≈5) | 0.035 | beats 61% (random-equivalent) | 4/8 |
| ours corrected (true 8 clusters) | **0.100** | beats 3% (**worse than random**) | **7/8** |
| stratified-random (our allocation + weights, random within cluster) | 0.054 | ≈ random | 7/8 |

Two findings:

1. **Script bug** — `run_scenario_e2e.py` re-clustered the pool at default k (≈5) when selecting
   K=8 instead of reusing the pool's 8 build clusters, so the shipped selection covered only 4/8
   intents. (`build_scenario_set` itself is correct; only the experiment script diverged.)
2. **Medoid bias, the real lesson** — for estimating the *pool-mean score*, the corrected,
   coverage-respecting selection is *worse* than random. The stratified-random control isolates
   the cause: the coverage allocation is neutral; deterministic intent-space **medoids are biased
   in score space** (per-scenario scores span 0.33–1.00, std 0.15), and 8 deterministic picks
   can't average that bias away, while uniform random is unbiased by construction. This matches
   the benchmark-compression literature — item-space representativeness ≠ score-space
   representativeness, which is exactly why IRT/model-aware selection (Design C) exists.

**Honest positioning:** this selection buys *coverage* (7/8 intent clusters vs ~5/8 random — a
regression in an untested intent is undetectable at any score) and *auditability* (provenance,
named clusters, pinned failures), not mean-score calibration. If mean-score prediction is the
goal: use random-within-cluster picks with the same allocation (unbiased, keeps coverage) or the
Design-C IRT pass. Follow-ups: fix the script; consider a `pick="medoid"|"random"` knob on
`hybrid_select` so the estimation use-case has an unbiased mode.

## Verdict

The pipeline runs end-to-end on real traces with a small model: construction produces named,
weighted, checklisted scenarios with provenance, and the verification loop produces actionable
per-scenario verdicts. Per the 2026-07-03 correction, the selection's measurable win is
**coverage and auditability, not mean-score calibration** — at K=8 it is random-equivalent (or
worse, when medoids bias the estimate) for predicting the pool-mean score. The concrete next
steps: unbiased within-cluster picking for the estimation use-case, a wider agent capability
spread for rank metrics, and a multi-rollout corpus for task-identity recovery.

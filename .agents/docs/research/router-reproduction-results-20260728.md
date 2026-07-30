# WMO router reproduction results

Status: Complete. Real evaluation and five-seed world-model replay are sealed.

Experiment: `router-real-wm-20260728`

Protocol: `.agents/docs/research/router-reproduction-20260728.md`

The deployment comparison is always router minus the best single model selected on fit data.
Heldout outcomes are used only for evaluation. A positive quality delta is better. A negative
cost delta is cheaper.

## Phase 1: real benchmark environments

| Benchmark | Best-single quality | Best-single cost per task | Router quality delta | Router cost delta | Promotes |
| --- | ---: | ---: | ---: | ---: | --- |
| RouterBench | 94.55% | $0.00122 | +0.2809 points | +99.17% | No |
| Tau2 | 88.00% reward | $0.29537 | 0.0000 points | 0.00% | No |
| Terminal-Bench 2 | 50.65% reward | $0.69701 | -0.3670 points | -2.20% | No |

The baseline columns are five-seed means. The fit-selected model can differ by seed. Gradeability
was 10,789 / 10,791 cells on RouterBench, 180 / 180 on Tau2, and 795 / 801 on
Terminal-Bench 2.

RouterBench became slightly more accurate but nearly doubled cost at the frozen balanced dial.
Tau2 routed exactly like its fit-selected best single model, so it delivered no quality or cost
change. Terminal-Bench 2 saved 2.20 percent but lost 0.37 quality points; its paired 95 percent
quality interval was `[-5.38, +4.59]` points, so the noninferiority and all-seed gates failed.

These results do not reproduce a promotable router at the preregistered operating point.

## Phase 2: world-model replay

RouterBench replay is complete across five independently fit world-model indexes:

- 53,636 gradeable final cells
- 54,657 all-attempt rows
- $636.239535 measured Azure GPT-5.5 world-model inference
- 96.51% binary cell agreement and 3.49-point mean absolute error
- 0 / 5 best-single agreement, despite mean model-quality Spearman correlation of 0.933
- 26.12% selected-model agreement and 40.28% guard-gate agreement
- 99.61% mean coverage for the simulated route-versus-baseline comparison

The WMO-fitted RouterBench policy rejects promotion, matching the real deployment decision. At
the balanced dial it predicts a 0.225-point quality loss and a 108.78 percent cost increase. Its
routes do not match closely, however. Replaying the WMO-selected policy against real outcomes
loses 0.337 quality points and costs 12.12 percent more than the policy fit on real outcomes.

Tau2 replay is complete:

- 170 / 180 gradeable final cells and 208 all-attempt rows
- $130.820903 candidate inference plus $159.026555 Azure GPT-5.5 world-model inference
- 47.65% binary cell agreement and 53.72-point mean absolute error
- 0 / 5 best-single agreement
- 0% selected-model agreement and 72% guard-gate agreement

The WMO-fitted Tau2 policy rejects promotion, matching the real deployment decision. It predicts
the same simulated quality as its simulated best single and 1.74 percent lower cost, but only two
of five seeds pass the joint point-estimate gate. The agreement is not sufficient validation:
WMO chooses Fable as the baseline instead of the real matrix's Sonnet, and replaying its policy
on real outcomes loses 20.0 reward points and costs 149.45 percent more than the real-data policy.

Terminal-Bench 2 replay is complete across five independently fit world-model indexes:

- 2,370 / 4,005 gradeable final cells, 8,454 all-attempt rows, and 90 retry cohorts
- $1,697.567627 candidate inference plus $4,397.936240 Azure GPT-5.5 world-model inference
- 67.30% binary paired-cell agreement and 33.27-point mean absolute error
- 58.87% mean paired coverage of gradeable real cells
- 0 / 5 best-single agreement
- mean model-quality Spearman correlation of 0.060 and Kendall correlation of 0.023

Coverage is highly model-dependent. GPT-5.4-mini and GPT-5.5 have only about 1% to 8% paired
coverage in individual seeds, while several other models have roughly 54% to 91%. Comparing
model means across those different observed subsets is selection-biased. The full WMO analyzer
therefore cannot construct a valid fit-selected baseline, and the final comparison deliberately
falls back to cell-only mode. No simulated Terminal-Bench routing or promotion decision is
reported.

The simulated rewards for Tau2 and Terminal-Bench 2 are world-model judgments, not official
real-environment verifier results. Real and simulated rows were never pooled.

## Current-main reconciliation

The run was frozen from source commit `c3267f1f`. At finalization, current `origin/main` was
`60fd6a43`, nine commits ahead of the freeze. Its independently run real-default studies also
report a router null result on SWE, Tau2, and Terminal-Bench 2: a pinned single model dominates
or matches the routed operating points on each workload.

That is directionally corroborating evidence, not part of this experiment. The current-main
studies use different model rosters, task samples, repetition counts, anchors, and analysis
procedures. Their rows and costs are not pooled with the frozen matrices above.

## Decision

The reproduction does not support promotion.

The real-environment gate is closed on all three benchmarks. RouterBench nearly doubles cost for
a statistically non-significant 0.28-point quality increase. Tau2 produces no change. Terminal-
Bench 2 trades a statistically uncertain 0.37-point quality loss for a 2.20 percent cost saving.

WMO is not validated as a replacement for real evaluation. It is strong at reproducing individual
RouterBench correctness labels and broad model ordering, but it selects the best single model in
0 / 5 seeds and its selected routes agree only 26.12 percent of the time. It fails much more
directly on Tau2 and cannot support a routed-policy estimate on Terminal-Bench 2 because coverage
is sparse and uneven.

The final comparison artifact is `cell-only`. That is an intentional scientific result, not a
missing analysis: a global WMO promotion-decision agreement would be misleading when one of the
three benchmarks cannot support policy fitting.

## Spend

The append-only ledger records $7,582.584512 of measured spend across accepted runs, excluded
paid attempts, replay, and smoke work. Three environment-side entries remain explicitly unknown
because the source artifacts contain no invoice amount. Phase 2 world-model replay accounts for
$7,021.590861 of the measured total:

- RouterBench: $636.239535
- Tau2: $289.847458
- Terminal-Bench 2: $6,095.503867

No world-model replay attempt has a metering gap.

## Evidence

All raw matrices, attempts, telemetry, analyses, and the spend ledger are stored outside Git at:

`/Users/admin/Documents/experientiallabs/data/router-repro-20260728/full`

The canonical final comparison is:

`/Users/admin/Documents/experientiallabs/data/router-repro-20260728/full/analysis/comparison.json`

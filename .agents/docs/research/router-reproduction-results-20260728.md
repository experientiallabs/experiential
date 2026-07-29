# WMO router reproduction results

Status: Phase 1 complete. Phase 2 simulations in progress.

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

Terminal-Bench 2 is still running. Its simulated rewards, like Tau2's, are world-model judgments,
not official real-environment verifier results. Real and simulated rows will never be pooled.

Terminal-Bench 2 has already exposed a fidelity limitation: many candidate episodes exhaust the
response output cap before acting, and scored simulated outcomes depend on the world model
deciding that the task is complete. Final claims must therefore include paired-cell coverage by
model and may reject the simulated deployment decision as unavailable if coverage is inadequate.

## Decision

Pending the sealed Phase 2 comparison. Regardless of simulation results, the real-environment
promotion gate is currently closed on all three benchmarks.

## Evidence

All raw matrices, attempts, telemetry, analyses, and the spend ledger are stored outside Git at:

`/Users/admin/Documents/experientiallabs/data/router-repro-20260728/full`

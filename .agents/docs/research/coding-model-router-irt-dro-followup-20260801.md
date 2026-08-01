# Conditional graded IRT router follow-up

Status: conditional preregistration draft. This is not an active experiment and does not change
the frozen graded SWE-rebench v47 kNN protocol. DeepSWE outcomes remain sealed.

## Trigger and evidence boundary

This lane may start only if the current graded kNN study fails an external promotion gate. The
failure point determines the valid confirmation source:

1. If kNN fails on development before the current 320-task confirmation is opened, this study may
   use the completed development matrix for fitting and the still-sealed confirmation exactly
   once.
2. If kNN reaches and opens the current confirmation, those outcomes become development evidence
   after the kNN decision. A new repository-disjoint external confirmation cohort must then be
   frozen before this study is fit. The opened confirmation may not be reused as final evidence.

No DeepSWE outcome, target-derived threshold, or target rerun is permitted. Only the first
latency-neutral policy that passes a fresh external confirmation may receive one DeepSWE transfer.

## Hypothesis

Guarded kNN treats every nearby task as local evidence but does not explicitly represent the main
structure in this matrix: five ordered reasoning efforts for one model plus one frontier guard.
A graded item-response model may generalize better by jointly estimating task difficulty, task
discrimination, and arm ability from fail-to-pass counts. A distributionally robust decision rule
may then protect the quality constraint under repository shift without an inference-time model
call.

This combines two relevant ideas:

- IRT-Router models model ability and query attributes explicitly:
  `https://arxiv.org/abs/2506.01048`.
- RACER applies a KL-divergence uncertainty set to reasoning allocation under distribution shift:
  `https://arxiv.org/abs/2605.10805`.

POLLINATOR's graph plus IRT predictor is a secondary ablation, not the primary family:
`https://openreview.net/forum?id=N59cvpjnlo`. Online contextual-bandit routers are excluded because
the one-shot target transfer provides no legitimate target feedback.

## Frozen candidate family

The primary outcome is the exact binomial pair `(f2p_passed, f2p_total)`, not a binary resolve flag
or an unweighted reward fraction. For task `i` and arm `a`, fit

`f2p_passed[i,a] ~ Binomial(f2p_total[i], sigmoid(q[i] dot theta[a] - b[i]))`.

The pre-call task representation produces:

- scalar difficulty `b[i]`;
- nonnegative discrimination vector `q[i]`;
- one ability vector `theta[a]` per model by reasoning-effort arm.

Candidate representation dimensions are 2, 4, and 8. The task encoder candidates are signed
hashing at 512 and 2,048 dimensions, the frozen prompt-shape vector, and their concatenation.
Regularization strengths are 0.1, 1, 10, and 100. Compare unconstrained arm abilities with one
variant whose Luna capacity coordinate is monotone from low through max. The Sol guard remains a
separate arm and receives no ordinal constraint.

The graph ablation constructs a fit-only task similarity graph from the same pre-call features,
propagates latent difficulty with graph Laplacian penalties 0.01, 0.1, and 1, and changes no online
feature contract. Repository identity, model output, patch, tests, verifier details, and future
trajectory content remain forbidden features.

## Implementation starting point

Reuse the tested optimizer and grouped-CV structure in
`.agents/scripts/coding_model_router_codeforces_irt.py`. Do not reuse its scientific assumptions
unchanged. The graded SWE-rebench adaptation must:

1. replace its equally weighted fractional cross-entropy with the exact binomial likelihood so a
   1-of-1 score does not carry the same information as a 100-of-100 score;
2. generalize the scalar difficulty and discrimination to the frozen latent dimensions above;
3. replace direct linear or Chebyshev scalarization with the repository KL-robust selection rule;
4. remove fitted arm abilities and all other coefficients from persisted reports;
5. freeze only task identity, selected arm, input hashes, route provenance, and aggregate fit
   diagnostics in a label-free route manifest;
6. preserve its finite-difference gradient test, grouped split assertions, shuffled-label control,
   and latency audit.

The existing Codeforces implementation remains an historical experiment and is not modified by
this follow-up.

The prepared numeric core lives in
`.agents/scripts/coding_model_router_graded_irt_core.py`. It implements the exact count-weighted
binomial likelihood, analytic gradients, multidimensional nonnegative discrimination, a pre-call
feature-conditioned variant that can score unseen tasks, and the forward-KL repository robust
lower bound. It also implements the frozen monotone-capacity variant with a differentiable
cumulative-softplus parameterization on the first latent coordinate for Luna low through max; the
sixth Sol arm remains unconstrained. Its inline tests cover finite-difference gradients for both
ability variants, exact denominator weighting, unseen-task prediction, monotone Luna ordering, and
the KL solution. The module has no filesystem or serialization surface. It remains conditional
infrastructure and does not activate this lane.

The pure protocol helpers in `.agents/scripts/coding_model_router_graded_irt_protocol.py` implement
seed-sensitive repository-disjoint folds with exact one-fold task coverage and a shuffled-label
control that permutes complete outcome rows only within repositories. These replace the older
Codeforces implementation's unseeded folds and corpus-wide shuffle. They load no outcomes, fit no
model, and persist no state, so preparing them does not activate this lane.

All fitting, cross-validation, bootstrapping, and latency measurement run on E2B or Azure. The Mac
only orchestrates and validates bounded artifacts. No foundation model, task embedding bank, or
fitted numeric router state is persisted. The remote worker may retain coefficients only for its
bounded fit process, freeze deterministic label-free route manifests and aggregate reports, then
destroy the worker and its coefficients.

## Robust selection rule

Use five repository-grouped outer seeds and five repository-grouped inner folds. Fit on inner
training repositories and predict every arm's graded pass probability on inner validation tasks.
For each candidate, enumerate cost penalties on the frozen grid `0, 0.005, 0.01, 0.02, 0.03`.

For every task, choose the least expensive arm whose repository-robust lower reward estimate is at
least 95 percent of the fit-selected static guard estimate. The robust estimate minimizes expected
reward over a KL-divergence ball around the inner-fold repository distribution. Radius candidates
are `0, 0.01, 0.03, 0.05, 0.1`. Radius and cost penalty are selected only from inner out-of-fold
predictions.

The mechanical winner minimizes cost subject to all of these fit-only conditions:

1. at least 95 percent quality retention in every outer seed;
2. at least 40 percent cost savings in every outer seed;
3. positive matched task-blind reward advantage;
4. positive within-repository shuffled-label advantage;
5. no static arm dominates it in both quality and cost;
6. no repository with at least five tasks loses more than 0.10 absolute reward;
7. route latency below 5 ms p50 and 20 ms p95 over 10,000 single-core decisions;
8. zero inference-time network calls.

Tie breaks are higher worst-seed retention, higher matched-blind advantage, lower latency, smaller
ephemeral coefficient count, lower latent dimension, and the frozen grid order.

## Required ablations

Report every static arm, task-blind effort mixing at identical arm traffic, cost-only routing,
seeded random routing, within-repository shuffled outcomes, guarded kNN, unconstrained graded IRT,
monotone-capacity IRT, graph-regularized IRT, robust radius zero, the selected robust radius, pair
oracles, and the full oracle.

An IRT result is not a routing gain unless it beats matched task-blind mixing with a positive
repository-bootstrap lower bound. Better calibration or latent interpretability alone cannot
promote it.

## External confirmation and target transfer

Refit the selected configuration ephemerally on external development, freeze its deterministic
label-free confirmation route manifest, then destroy the fitted state before opening confirmation
outcomes. Use 10,000 repository-cluster bootstrap draws with seed 20260801. Promotion requires at
least 95 percent quality retention, at least 40 percent savings, a nonnegative lower bound for
`router_reward - 0.95 * fit_selected_static_reward`, positive matched-blind advantage, no static
dominance, and route p95 below 20 ms.

Only after that report and every source hash pass may the policy produce a label-free DeepSWE route
manifest. Evaluate that manifest exactly once against the sealed graded DeepSWE matrix. Never tune,
repair, or rerun from target outcomes.

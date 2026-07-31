# BigCodeBench reasoning-effort transfer experiment

Date: 2026-07-31

Status: preregistered, current-model outcomes unopened

## Question

Can a latency-neutral router learn when `gpt-5.6-luna` needs more reasoning effort from a fast,
execution-scored external coding benchmark, then transfer that frozen decision rule to DeepSWE
v1.1 without fitting on DeepSWE outcomes?

This experiment changes only reasoning effort within one model family. It makes no inference-time
router model call and persists no foundation-model weights.

## External source

- Dataset: `bigcode/bigcodebench`
- Dataset commit: `b74c0d0bf70d2c0bc459be537895cca163007f1a`
- Dataset split: `v0.1.4`
- Hard subset: `bigcode/bigcodebench-hard`
- Hard-subset commit: `298d2cc7b96612e15e47313c3603ee124cee0c1f`
- Evaluator: `bigcode-project/bigcodebench` release `v0.2.4`
- Evaluator commit: `9059fb84d1188c02edeac4995361656a2fdecbef`

The public release archive was inspected only to confirm feasibility. It contains 118
temperature-zero, one-sample model arms over the external tasks. Those historical model outputs
are not labels for this experiment.

## Frozen task cohort

The cohort contains 300 BigCodeBench Instruct tasks selected without current-model outcomes:

1. Load the pinned `v0.1.4` full and hard task tables.
2. Remove any exact task-id or normalized-prompt overlap with the existing label-free DeepSWE
   feature view. Access to DeepSWE rewards and costs remains forbidden.
3. Include every retained hard-subset task.
4. Fill the cohort to 300 from the remaining full tasks by ascending
   `sha256("20260731:" + task_id)`.
5. Persist the ordered ids, normalized task-family groups, source hashes, and cohort hash before
   the first provider call.

The task-family group is the sorted library signature from the dataset metadata. Missing library
metadata receives its own explicit group. Family groups, not individual rows, are the resampling
and cross-validation unit.

## Frozen arms and attempts

| Arm | Provider model | Reasoning effort |
|---|---|---|
| `luna-low` | `gpt-5.6-luna` | `low` |
| `luna-medium` | `gpt-5.6-luna` | `medium` |
| `luna-high` | `gpt-5.6-luna` | `high` |
| `luna-xhigh` | `gpt-5.6-luna` | `xhigh` |
| `luna-max` | `gpt-5.6-luna` | `max` |

Run five independent provider calls per task and arm, for 7,500 cells. Omit sampling temperature
because the provider's reasoning interface does not expose it for this model. Each call receives
the official Instruct prompt plus a fixed instruction to return only the Python implementation.
The output ceiling is 32,768 tokens for every effort.

## Scoring and failure policy

- Score with the pinned official BigCodeBench evaluator on remote E2B compute.
- A completed response that is empty, malformed, truncated, times out during tests, or fails tests
  is a gradeable zero.
- Retry only pre-response provider transport failures and evaluator infrastructure failures.
- Use at most five provider retries with bounded exponential backoff.
- Persist every completed call, raw response hash, sanitized code hash, usage, cost provenance,
  latency, and score immediately.
- Clear provider credentials from the execution subprocess environment.
- Run generation, execution, bootstraps, and fitting remotely. The local Mac is limited to code
  editing, orchestration, compact artifact sync, and lightweight tests.

The shared hard ceiling remains USD 20,000. Valid trace-derived spend before this experiment is
USD 14.5822, plus two ungradeable failed calls without valid cost accounting. Reserve USD 0.50 per
pending cell and stop before the shared ceiling could be exceeded. Exact provider telemetry is
preferred; otherwise label the trace-derived estimate.

## Held-out-attempt oracle gate

Before fitting any router, evaluate all ten exact choices of two fit attempts and three held-out
attempts:

1. Select the best effort for each task using only the two fit attempts.
2. Score that choice on the other three attempts.
3. Select the best static effort using only fit attempts.
4. Score the static effort on the same held-out attempts.
5. Measure held-out oracle reward minus held-out static reward.

Combine attempt splits with 2,000 task-family bootstrap resamples. Proceed to router fitting only
when every condition passes:

1. At least 250 uncontaminated tasks remain.
2. All five arms have five gradeable attempts on the same task cohort.
3. Mean held-out oracle headroom is at least 0.10 absolute reward.
4. The combined family-and-attempt 95 percent interval lower bound exceeds 0.05.
5. No arm or attempt is silently dropped.

If this gate fails, preserve the negative result and do not fit a router or open DeepSWE outcomes.

## Frozen latency-neutral router search

This search space is frozen before any current-model BigCodeBench reward is computed. It borrows
the efficient prompt-only supervision idea from RouteLLM, the task-profile prior from TRouter,
the counterfactual objective from doubly robust policy learning, and the locality and feature
weighting ideas behind guarded kNN and adaptive clustering. ACRouter's execution-feedback loop is
excluded because it adds calls and changes the single-call serving contract.

Every candidate receives only data available before inference:

- signed character n-gram hashes at 512, 2,048, and 8,192 dimensions;
- deterministic prompt-shape features such as length, lines, imports, type annotations, examples,
  tests, exceptions, recursion, and library count;
- the frozen BigCodeBench library signature and hard-subset indicator;
- prompt-family centroids and statistics fitted only inside the current training fold.

External embedding APIs, language-model classifiers, generated task descriptions, response
probes, self-consistency samples, cascades, verifier feedback, and target outcomes are forbidden.
The served artifact may contain only deterministic feature parameters and small fitted numeric
arrays. It must make no network call, persist no foundation-model weights, and route in less than
5 ms p50 and 20 ms p95 on one E2B CPU core over at least 10,000 repeated decisions.

The preregistered candidate families are:

1. **Guarded local kNN.** WMO reward-profile kNN over each frozen representation. Search 8, 16,
   32, and 64 neighbors; relative similarity 0.90, 0.95, and 0.98; guard z 0, 0.5, 1.0, and
   1.645; and minimum paired support 8, 16, and 32. Weak or novel neighborhoods revert to the
   fit-selected static effort.
2. **Ordinal adjacent-effort uplift.** Cross-fitted Ridge and ExtraTrees heads predict the four
   adjacent gains from low through max. Ridge alpha is 0.1, 1, 10, or 100. ExtraTrees uses 200 or
   500 trees, leaf size 5, 10, or 20, and at most square-root or one-third of features per split.
   Isotonic projection makes predicted cumulative reward nondecreasing in effort. The policy picks
   the cheapest effort whose lower confidence bound clears the fit-selected quality floor.
3. **Multi-action doubly robust policy.** Group-cross-fitted direct reward heads and known uniform
   arm propensities form augmented inverse-propensity pseudo-values for all five efforts. The
   policy learner is Ridge or histogram gradient boosting with maximum leaf nodes 7, 15, or 31,
   learning rate 0.03 or 0.10, and minimum leaf size 10 or 20. A fit-only shadow price chooses
   reward minus lambda times each arm's mean fit cost divided by the across-arm mean fit cost.
   This gives lambda units of reward points per average call, matching WMO's `pick_lam` scale.
   Lambda is 0, 0.0025, 0.005, 0.01, 0.02, or 0.04.
4. **Empirical-Bayes family shrinkage.** Beta-binomial task and library-family effects shrink
   repeated binary executions toward global and hard-subset priors. Global arm means receive
   Jeffreys smoothing before they define the empirical prior, preventing zero posterior variance
   on all-pass or all-fail arms. A Ridge residual head predicts remaining adjacent-effort uplift
   from the same frozen representations. Prior strength is 2, 5, 10, 20, or 50 effective trials.
   Posterior lower bounds use z 0, 0.5, 1.0, or 1.645 and revert to the fit-selected static effort
   when no effort clears the fit-only quality floor.

The Cartesian grids contain 432 guarded kNN points and 576 non-kNN points, for 1,008 candidates
per outer seed before negative controls. Candidate identities and frozen ordering are generated
without reading a reward row.

All fitting and hyperparameter search runs remotely. The five outer seeds are 0 through 4. For
each seed, complete library signatures are ordered by
`sha256("bigcodebench-outer-v1:<seed>:<group>")`. The held-out partition is the shortest prefix
whose task count is closest to 20 percent, comparing the counts immediately before and after the
target and choosing the longer prefix on an exact tie. The remainder is fit data. This split uses
task identities and library signatures only, before any reward access. All five attempts for a
task stay in one fold. Inner folds select the
least-cost point satisfying the 95 percent quality floor. When no point clears the floor, the
fallback maximizes fit-only quality, then prefers lower cost, lower latency, smaller artifact, and
frozen candidate order. The outer comparison includes every static effort,
matched task-blind effort mixtures, shuffled-label policies, cost-only routing, random routing,
unguarded versions of each family, and the held-out-attempt oracle. Candidate selection uses the
mean across five deterministic outer seeds, with ties resolved by lower cost, lower route latency,
smaller artifact, and then the order above.

The frozen task manifest produces the following outer partitions. Each digest is SHA-256 of the
ordered held-out task ids, one id per line with a trailing newline:

| Seed | Fit tasks | Held-out tasks | Held-out groups | Held-out digest |
|---:|---:|---:|---:|---|
| 0 | 240 | 60 | 41 | `ab6debce045572f312cb0d7ff65d2ce5db6ab1c1b1d3e58c549fbf1d98b75ee4` |
| 1 | 240 | 60 | 44 | `33c72cf3fafe6834ab721c932e5a73770b5e554c0c194b3578d22b99102abcdf` |
| 2 | 240 | 60 | 50 | `1755c92a082b0fc7759638e187adbfefb27a983c19663f4fae553dac833e31dc` |
| 3 | 240 | 60 | 42 | `0d1ae009fe32a42f63c648d74189bf208802e48af7a2892e1137a0209804a5a4` |
| 4 | 241 | 59 | 45 | `24a30930a802192f3ecd4aa584b60b7016b95c0879d13486ae93fbcde7638255` |

Primary references:

- RouteLLM: `https://arxiv.org/abs/2406.18665`
- Doubly Robust Policy Evaluation and Learning: `https://arxiv.org/abs/1103.4601`
- TRouter: `https://arxiv.org/abs/2604.09377`
- Adaptive Clustering router: `https://arxiv.org/abs/2502.15315`
- ACRouter and CodeRouterBench: `https://arxiv.org/abs/2606.22902`

## Router promotion gate

If the oracle passes, fit and tune only on external outcomes using nested family-grouped
cross-validation. Candidate policies may use only pre-call task text and metadata. The primary
selection rule is the least expensive policy that, across five deterministic outer seeds:

1. retains at least 95 percent of the fit-selected strongest static arm's held-out quality;
2. saves at least 40 percent held-out cost;
3. has a paired 95 percent interval that does not exceed the allowed five percent relative loss;
4. has no task-family catastrophic regression hidden by the aggregate;
5. beats task-blind, shuffled-label, random, cost-only, and static-effort controls.

Only one externally selected policy and operating point may advance. Freeze its feature transform,
fitter, hyperparameters, cost-quality dial, and artifact hash before reading target outcomes.

## DeepSWE transfer

DeepSWE v1.1 remains evaluation-only. If the external promotion gate passes:

1. Apply the frozen policy to the existing label-free DeepSWE feature view.
2. Map its selected Luna effort directly to the matching DeepSWE arm.
3. Evaluate once against fit-selected static and matched task-blind controls using graded
   fail-to-pass reward, measured cost, repository-grouped uncertainty, and model-times-effort arm
   identity.
4. Report the result as adaptive target transfer, not untouched confirmation.

No target feature search, threshold tuning, cost-penalty tuning, or repeated replay is permitted.

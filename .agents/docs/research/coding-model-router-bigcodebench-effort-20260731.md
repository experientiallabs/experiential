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

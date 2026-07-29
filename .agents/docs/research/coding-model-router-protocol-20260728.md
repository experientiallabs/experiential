# Execution-scored coding-model router protocol

Status: frozen before the first paid benchmark cell.

Experiment ID: `coding-router-20260728`

Source commit: `c3267f1f9d5f35a14ad45b6a94b7b21d3b11c958`

Branch: `exp/coding-model-router-20260729`

Worktree:
`/Users/admin/Documents/experientiallabs/.codex/worktrees/world-model-optimizer/coding-model-router-20260729`

Material paid sweep ceiling: not authorized.

## Objective

Build and serve the least expensive pre-inference WMO routing policy that retains at least 95
percent of the held-out quality of the strongest OpenAI or Anthropic static model selected on fit,
while saving at least 40 percent of its realized inference cost. Prefer at least 60 percent savings
when the same quality and statistical gates hold.

## Scientific gates

1. Select the static frontier baseline using fit rows only.
2. Held-out relative quality retention must be at least 0.95.
3. Held-out realized inference cost must be at least 40 percent below the baseline.
4. Report relative retention and absolute percentage-point quality delta.
5. Use paired scenario-level 95 percent confidence intervals.
6. No benchmark may lose more than 10 relative quality points or 10 absolute points.
7. Apply the point-estimate gates independently on split seeds 0 through 4.
8. Promotion requires all five seeds to pass plus the pooled paired interval gate.
9. The production choice is the least expensive preregistered point that passes.
10. Held-out results may not change the search space, dial grid, or selection rule.

## Benchmarks

| Cohort | Execution pin | Primary reward | Split group |
| --- | --- | --- | --- |
| Terminal-Bench 2 | Harbor `terminal-bench@2.0`, 89 tasks, `terminal-bench-2` commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` | official Harbor verifier | task family |
| SWE-bench Verified | Harbor `swebench-verified@1.0`, 500 tasks, `harbor-datasets` commit `86723674f04e4209ac479d0fb75d9d9f44b4377e` | official repository tests | repository |

The SWE-bench execution identities are cross-checked one for one against
`princeton-nlp/SWE-bench_Verified` commit
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. The frozen parquet has SHA-256
`a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`.

LiveCodeBench is excluded from the primary protocol because this source revision has no reliable
execution-scored WMO adapter. It may be added only as a separately labeled future cohort.

The aggregate gives each benchmark weight 0.5, regardless of task count. Per-benchmark results
remain primary.

## Splits and leakage controls

Seeds are 0, 1, 2, 3, and 4. A deterministic SHA-256 ordering and subset optimizer chooses whole
groups closest to the 70 percent fit target. Every seed is 62/27 for Terminal-Bench 2 and 350/150
for SWE-bench Verified. All seeds are unique, every task appears exactly once, and no group crosses
fit and held-out.

Router features contain only the task statement and metadata available before the first model
call. Patches, hidden tests, verifier output, reward, future tool trajectory, and held-out labels
are excluded.

Frozen artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Terminal-Bench 2 manifest | `7f59be4fbcc715fcbadf998ccf3d11f00f7fc9c978b196407e0c051c76f6b835` |
| SWE-bench Verified manifest | `6c773fc71e8bda17bb907d2f309c50d3f1b5d887dfda8a6841df066d2f2eed79` |
| Model pool | `b909fe11bc1c89d6e6501d9b750b468aa4abdfab5e389357980608049f99663e` |
| Seed 0 | `83b2a7334a0fa57914b12769d173d78deffb8f84397f18d3accd83a32c6c81e2` |
| Seed 1 | `211a0e1564958691f52a753e1478a58b95acfb4dd2717856bedf6f03a0ca0d3c` |
| Seed 2 | `8d182367772b21d15e59795c7b6eee47c89b39dc77e34520400448121470beb3` |
| Seed 3 | `d8abbbd5d6a451146842b4739c79d024c5f9700ca3286ffcce7ba89c63025716` |
| Seed 4 | `eff42456f5ddd386f31a6b4a4263bd6934bcc3a7637453f0cb90abf87297120b` |

## Candidate roster

| Arm | Exact provider model | Effort | Standard input/cache/output USD per million |
| --- | --- | --- | --- |
| `oai-sol-max` | `gpt-5.6-sol` | max | 5.00/0.50/30.00 |
| `oai-sol-high` | `gpt-5.6-sol` | high | 5.00/0.50/30.00 |
| `oai-terra-max` | `gpt-5.6-terra` | max | 2.50/0.25/15.00 |
| `oai-terra-high` | `gpt-5.6-terra` | high | 2.50/0.25/15.00 |
| `oai-luna-high` | `gpt-5.6-luna` | high | 1.00/0.10/6.00 |
| `oai-gpt55-high` | `gpt-5.5-2026-04-23` | high | 5.00/0.50/30.00 |
| `oai-codex53-high` | `gpt-5.3-codex` | high | 1.75/0.175/14.00 |
| `oai-mini54-high` | `gpt-5.4-mini-2026-03-17` | high | 0.75/0.075/4.50 |
| `ant-fable-max` | `claude-fable-5` | max | 10.00/1.00/50.00 |
| `ant-opus5-max` | `claude-opus-5` | max | 5.00/0.50/25.00 |
| `ant-opus5-high` | `claude-opus-5` | high | 5.00/0.50/25.00 |
| `ant-sonnet5-high` | `claude-sonnet-5` | high | 3.00/0.30/15.00 |
| `ant-sonnet5-low` | `claude-sonnet-5` | low | 3.00/0.30/15.00 |
| `ant-haiku45` | `claude-haiku-4-5-20251001` | off | 1.00/0.10/5.00 |

Each exact ID was returned by its provider's live read-only model-list API on 2026-07-28. Sonnet 5
uses the standard 3/15 price rather than a temporary introductory rate.

Exact model capabilities frozen from the providers' official model pages:

| Exact model | Context | Max output | Structured tool use | Availability at freeze |
| --- | ---: | ---: | --- | --- |
| `gpt-5.6-sol` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.6-terra` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.6-luna` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.5-2026-04-23` | 1,050,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.3-codex` | 400,000 | 128,000 | Responses function calling | live model list |
| `gpt-5.4-mini-2026-03-17` | 400,000 | 128,000 | Responses function calling | live model list |
| `claude-fable-5` | 1,000,000 | 128,000 | Messages tool use | live model list |
| `claude-opus-5` | 1,000,000 | 128,000 | Messages tool use | live model list |
| `claude-sonnet-5` | 1,000,000 | 128,000 | Messages tool use | live model list |
| `claude-haiku-4-5-20251001` | 200,000 | 64,000 | Messages tool use | live model list |

OpenAI publishes a Tier 1 floor of 500 requests/minute and 500,000 tokens/minute for these
roster entries; higher account tiers vary by model. Anthropic's published Start tier is
1,000 requests/minute for every Claude entry, with 500,000 input and 100,000 output
tokens/minute for Fable 5, and 2,000,000 input and 400,000 output tokens/minute for Opus 5,
Sonnet 5, and Haiku 4.5. Account and workspace overrides may be lower. The experiment therefore
starts at four concurrent cells, observes response rate-limit headers, obeys `retry-after`, and
never interprets a pre-execution 429 as a gradeable model failure.

GPT-5.5 and GPT-5.6 calls above 272,000 input tokens are priced per request at the documented
long-context tier: 2x input and cached-input rates plus 1.5x output rates. The runners persist
per-call input, cached-input, cache-write, and output counters because an episode aggregate
cannot determine that tier.

Official sources:

- `https://developers.openai.com/api/docs/models`
- `https://developers.openai.com/api/docs/models/gpt-5.5`
- `https://developers.openai.com/api/docs/models/gpt-5.3-codex`
- `https://developers.openai.com/api/docs/models/gpt-5.4-mini`
- `https://platform.claude.com/docs/en/about-claude/models/overview`
- `https://platform.claude.com/docs/en/api/rate-limits`

## Attempts, retries, and evidence

The primary matrix uses one model attempt per task with the same harness, turn cap, output cap,
wall timeout, and official verifier. Gradeable model failures are never retried. Only a missing
environment, sandbox failure, transport loss before execution, or missing verifier reward is an
infrastructure failure. It receives at most two fresh-sandbox retries after 15 and 60 seconds.

Every attempt is retained. Every completed cell atomically persists reward, success, tokens,
cache reads and writes, reasoning tokens, realized model cost, per-call latency, wall time, tool
calls, stop reason, completion status, failure class, attempt number, and raw Harbor artifact path.
Environment duration is recorded even when E2B does not expose its invoice rate.

## Single smoke gate

The only paid pre-sweep gate has exactly four cells:

- fit task `break-filter-js-from-html`;
- held-out task `log-summary-date-ranges`;
- `oai-luna-high`;
- `ant-haiku45`.

Each cell runs separately through Harbor with an E2B task environment and official verifier, then
persists before the next cell starts. After two cells the runner intentionally exits and is
resumed. The resumed run must retain byte-identical completed artifacts, finish the other two
cells, fit a guarded hashing-1024 kNN plumbing policy on the fit task, and replay the held-out task.
The smoke has a hard USD 10 inference cap and is not headline evidence.

The no-spend task and provider preflight passed for all four cells. The paid gate is blocked on
shared E2B capacity. At 2026-07-29 00:07 PDT the account reported 226 running sandboxes against
the configured cap of 100, zero slots free, and no provably orphaned local sandbox eligible for
safe reaping. A fresh check after the replacement worktree was created still returned a full
100-sandbox first page plus additional pages. The experiment will not override the shared cap or
kill account-wide work owned by another process.

## Router search

The production family is WMO guarded kNN. The frozen grid covers:

- hashing-1024 and direct OpenAI `text-embedding-3-large` at native 3072 dimensions;
- neighbors 8, 16, 32, and 50;
- relative similarity threshold 0.90, 0.95, and 0.98;
- novelty quantile 0, 0.05, 0.20, and 0.50;
- guard z 0, 0.5, 1.0, and 1.645;
- minimum paired evidence 3, 5, 8, and 12;
- standard-error floor off and on;
- asymmetric cost guard off and on;
- cost-quality dial 0, 0.10, 0.25, 0.50, 0.75, and 1.0;
- benchmark-stratified bank off and on;
- missing-cell minimum coverage 0.8 and 1.0.

Baselines are every static arm, fit-selected best single, cheapest single, seeded random, cost
only, unguarded kNN, guarded kNN, rank routing, and oracle per-task routing. Cascades and retry
escalation are research-only policies and cannot be the production choice.

### Fit-only selection and heldout lock

Hyperparameters are selected without touching any outer heldout reward:

1. Within each outer seed's fit partition, assign whole repositories and task families to five
   inner folds by SHA-256 of
   `inner-v1:<outer-seed>:<benchmark>:<group>`. The same group never appears in inner train and
   validation.
2. Select the static frontier baseline separately on each outer fit partition using the declared
   0.5/0.5 benchmark aggregate. Ties go to lower realized fit cost, then frozen pool order.
3. Start from hashing-1024, 50 neighbors, threshold 0.95, novelty quantile 0.05, z 0.5,
   minimum eight pairs, standard-error floor on, symmetric guard, and dial 0.25.
   In this factorial search, the dial coordinate contributes its native WMO `pick_lam` cost
   pressure while the separately searched novelty, z, and guard coordinates remain authoritative.
   This avoids silently overwriting three earlier coordinates when the dial is visited. The
   deployable artifact records both the dial label and the effective primitive knobs. Serving
   verification separately exercises WMO's standard live dial mapping.
4. Run two deterministic coordinate passes in this order: embedder, neighbors, similarity
   threshold, novelty quantile, guard z, minimum pairs, standard-error floor, asymmetric guard,
   cost-quality dial. Run this search independently inside each outer seed. At each coordinate,
   evaluate every preregistered value by that seed's five-fold inner validation only. A seed's
   configuration may never use another seed's fit rows because those rows can overlap its own
   outer heldout set.
5. A coordinate value is feasible only when that outer seed's inner-validation aggregate retains
   at least 95 percent of its fit-selected baseline and passes both per-benchmark
   catastrophic-regression limits. For each of Terminal-Bench 2 and SWE-bench Verified, quality
   retention must be at least 0.90 and absolute quality loss must be no more than 0.10. Pick the
   lowest-cost feasible value. If none is feasible, maximize retention, then mean quality, then
   lower cost, then the frozen value order.
6. Benchmark-stratified banks and 0.8 missing-cell coverage are reported as one-at-a-time
   ablations from the selected point. They cannot replace the production point unless they were
   selected through the same fit-only rule.
7. Atomically write `selection-lock.json` with five independently selected hyperparameter sets,
   five fit-selected baselines, inner-validation metrics, matrix digest, split digests, code
   commit, and a deterministic deployment consensus before any outer-heldout policy replay. For
   each discrete coordinate, the consensus is the modal selected value; ties use the frozen value
   order.
8. Fit one outer policy per seed using that seed's locked configuration and full fit partition,
   then evaluate that seed's outer heldout exactly once. All preregistered static, dial, guard,
   rank, random, cost-only, and oracle points may be replayed for the locked Pareto report, but no
   heldout result may revise any seed configuration or the deployment consensus.
9. The deployable artifact refits the pre-heldout consensus hyperparameters on all real rows and
   pins the fit-only consensus baseline: majority of the five outer fit selections, with ties
   resolved by mean outer-fit quality, mean outer-fit cost, then frozen pool order. The headline
   heldout claim is the nested five-seed procedure, not a post-hoc evaluation of this all-row
   deployment refit.

## Statistics and world-model comparison

Confidence intervals use 10,000 paired scenario bootstrap resamples respecting repository and task
family clusters. Each resample preserves the 0.5/0.5 benchmark weighting. Promotion requires the
lower bound of the pooled paired retention interval to remain at or above 0.95, in addition to
all five split point-estimate gates. Reports include quality, cost, effective cost per success,
completion, gradeability, latency p50 and p95, model mix, route-away share, guard reversion,
novelty abstention, and declared capability slices.

After real matrices and splits are immutable, Azure GPT-5.5 world-model inference builds a separate
simulated matrix. Real and simulated rows are never pooled. Compare cell agreement, false positive
and negative rates, calibration, model rank, best-single choice, routed-model choice, guard
decision, predicted deltas, and final promotion decision.

The simulated environment is built from exactly one reward-free trajectory from the real
fit-selected deployment-consensus baseline for each task. Rewards and verifier labels are removed,
but task-specific observations remain retrievable, so the comparison measures reconstruction and
decision agreement rather than unseen-task generalization. Candidate actions in simulation use
WMO's native `LLMAgent`; the real matrix uses Harbor with the default Pi agent. That scaffold
difference is an explicit simulation-to-real confound and must be carried into the report.

## Serving gate

Mount the selected policy and evidence bank through `wmo serve`. Exercise unseen coding requests
through the OpenAI-compatible endpoint and verify both provider routes, multi-turn tool calls,
audit evidence, cost and cache accounting, conversation affinity, safe fallback, and the live
cost-quality dial.

## Spend and durability

No full benchmark cell may launch until a user-provided material paid sweep ceiling is recorded in
the freeze summary and enforced by the spend ledger. The bounded four-cell smoke is the only paid
work permitted before that ceiling.

Raw artifacts live under `.wmo/experiments/coding-router-20260728/` and stay out of Git. The
protocol and one-off runners live under `.agents/`. Long jobs use tmux with persistent logs. A
launch is accepted only after completed-cell counters advance on two successive polls.

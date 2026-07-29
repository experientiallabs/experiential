# WMO router real-to-world-model reproduction

Status: protocol frozen before any additional paid full-matrix call.

Experiment ID: `router-real-wm-20260728`

Source commit: `c3267f1f9d5f35a14ad45b6a94b7b21d3b11c958`

Branch: `exp/router-real-repro-20260728`

Hard combined spend ceiling: USD 20,000.

## Question

Does guarded kNN routing improve quality and reduce realized cost relative to the model selected
as best single on fit data, on RouterBench, real Tau2, and real Terminal-Bench 2? After the real
matrices are frozen, does an actual WMO world model served by Azure GPT-5.5 lead to the same
router and deployment decision?

Historical replay and refreshed current-model replication are separate results. The private
`routerbench-ours9` outcome matrix and cached `text-embedding-3-large` vectors are absent from the
repository and public storage. The zero-cost historical validator is run if those artifacts are
recovered. Otherwise, the historical headline is marked unavailable and the refreshed matrix is
the replication, not an exact numerical reproduction.

## Source pins and task cohorts

| Benchmark | Pin | Frozen cohort | Ground-truth reward |
| --- | --- | --- | --- |
| RouterBench | `withmartian/routerbench` public `routerbench_0shot.pkl`, SHA-256 recorded in the manifest | the historical Stage B procedure: certify MCQ answer keys from official score columns, then stratified seed 11 sample, target 1,199 | exact answer letter correctness |
| Tau2 | upstream commit `1d244f5dca42944b67a379b44bfeb9f5748f189d` | checked-in 20-task heldout set: 7 airline, 8 retail, 5 telecom | official Tau2 episode reward |
| Terminal-Bench 2 | source commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`, Harbor `terminal-bench@2.0` | all 89 resolved tasks | official Harbor verifier reward |

Tau2 uses the intended heldout test band, not the 97 world-model training tasks. World-model
construction may use only those 97 training IDs. All current-model candidates run once per real
task. Gradeable failures and zero rewards remain in the matrix.

The real Tau runner must receive `freeze/tasks/tau2.json` through `--task-manifest`. Its legacy
corpus-derived `scenarios_eval.jsonl` differs on five telecom tasks and is not an admissible
substitute. Any cohort run from that file is preserved as an invalid attempt and excluded rather
than relabeled.

Five deterministic paired split seeds are 0, 1, 2, 3, and 4. Each split uses 70 percent fit and
30 percent heldout, with deterministic hash ordering. RouterBench is stratified by benchmark
prefix. Tau2 is stratified by domain. Terminal-Bench 2 is grouped by task family. No group crosses
fit and heldout for a given seed.

## Original roster and frozen substitutions

The exact nine-model Stage B roster was recovered from commit `d279e0b9` and
`.agents/scripts/build_dashboard.py`:

| Original name | Current arm and exact route | Price snapshot, USD per million input/cache/output | Difference and exactness |
| --- | --- | --- | --- |
| `gpt-5.5` | Azure OpenAI, deployment from `AZURE_FOUNDRY_GPT55_DEPLOYMENT` on resource 2 | 5.00/0.50/30.00 | same canonical model; deployment value is secret and recorded only by env variable name |
| `gpt-5.4-mini` | Azure OpenAI, deployment from `AZURE_FOUNDRY_GPT54_MINI_DEPLOYMENT` on resource 2 | 0.75/0.075/4.50 | same canonical model |
| `fable-5` | Anthropic `claude-fable-5` | 10.00/1.00/50.00 | same canonical model |
| `sonnet-5` | Anthropic `claude-sonnet-5` | 3.00/0.30/15.00 | same canonical model |
| `haiku-4-5` | Anthropic `claude-haiku-4-5-20251001` | 1.00/0.10/5.00 | same dated snapshot |
| `opus-4-8` | Bedrock `us.anthropic.claude-opus-4-8`, `us-east-1` | 5.00/0.50/25.00 | same canonical model |
| `deepseek-v4-pro` | Azure Foundry deployment from `AZURE_FOUNDRY_DEEPSEEK_DEPLOYMENT` on resource 1 | 1.74/no published cache discount/3.48 | same canonical model |
| `kimi-k2.6` | Azure Foundry deployment from `AZURE_FOUNDRY_KIMI_DEPLOYMENT` on resource 1 | 0.95/no published cache discount/4.00 | same canonical model |
| `glm-5.2` | Azure Foundry deployment from `AZURE_FOUNDRY_GLM52_DEPLOYMENT` on resource 3 | 1.54/0.15/4.84 | same canonical model; Azure Retail Prices meters `FW GLM 5.2 Inp DZ`, `Cache Inp DZ`, and `Outp DZ`, queried 2026-07-29 |

If an exact route fails its availability check, the cell is not silently dropped. The failure is
persisted. A nearest replacement requires a protocol amendment before its first call, names the
capability difference, and makes the refreshed result non-exact for that arm.

Reasoning and agent settings are identical within each benchmark: high reasoning where supported,
native default thinking where an API does not expose the same knob, 100 agent turns for Tau2, 20
agent turns for Terminal-Bench 2, and 1,024 output tokens for RouterBench exact-answer calls.

## Router policies and baselines

Every seed selects the best single model on fit rows only. Router fitting sees only fit task text
and fit outcomes. Heldout outcomes are read only for the final paired evaluation.

The frozen promoted policy is WMO guarded kNN with:

- `text-embedding-3-large`, 3072 dimensions;
- 50 neighbors;
- relative similarity threshold 0.95;
- `guard_z=0.5`;
- symmetric paired guard;
- minimum paired evidence 8;
- standard-error floor on;
- fallback and guard model equal to the fit-selected best single.

The hashing-1024 representation is an explicit zero-cost representation ablation. The unguarded
ablation uses the same neighbor bank with `guard_z=0`, minimum pairs 0, and no statistical
reversion. Static baselines are every single model, fit-selected best single, cheapest single, and
seeded random. Oracle per-task routing is reported only as an upper bound.

Cost-quality dial points are 0, 0.25, 0.50, 0.75, and 1.00. They are applied after fitting without
refitting the evidence bank. The promoted quality point, shipped balanced point 0.25, all five
anchors, guarded, unguarded, and static best-single are reported on the same heldout IDs.

## Retries and missing cells

One gradeable attempt is the primary matrix row. Wrong answers, agent refusal, malformed tool use,
turn exhaustion, and any other run with an official reward are gradeable and never retried.

Only a missing environment, sandbox creation failure, transport loss before agent execution,
missing verifier output, or another failure with no official reward is infrastructure. It receives
at most two fresh-environment retries after 15 and 60 seconds. Every attempt is retained. If all
three attempts are ungradeable, the cell remains missing with its irrecoverable cause.

## Statistical and promotion rule

Primary quality is the benchmark's official reward. Report quality, realized model cost, model
plus environment cost where available, cost per attempt, cost per gradeable task, effective cost
per success, completion, gradeability, p50 and p95 latency, mix, route-away share, guard reversion,
and novelty abstention.

For every seed, report router minus fit-selected best single on identical heldout IDs. Confidence
intervals use 10,000 paired bootstrap resamples, clustered by RouterBench benchmark prefix, Tau2
domain, or Terminal-Bench family.

The router promotes only if all are true:

1. Mean pooled quality delta is nonnegative.
2. The paired 95 percent lower confidence bound is no worse than negative 0.5 percentage points.
3. Mean realized cost is lower.
4. Every benchmark loses less than 5 absolute quality points and less than 10 relative percent.
5. The same point-estimate gates pass on all five seeds.

The selected production point is the least expensive preregistered dial point that passes. This
rule and search grid cannot change after heldout results are observed.

## Single composite smoke record

Only one composite smoke is recognized. Work completed before this final protocol is folded into
that one gate, not treated as headline evidence:

- RouterBench: public matrix path and objective score path verified.
- Tau2: real official environment, tool execution, official reward, usage, persistence, and resume
  verified on the canonical heldout cohort.
- Terminal-Bench 2: Harbor plus E2B, official verifier, token and cost accounting, persistence, and
  resume verified in four minimal cells.
- Azure GPT-5.5 world model: exactly one smallest meaningful serve-and-step cell remains.

No other pilot or progressively larger preview may run. After the Azure cell passes, work proceeds
directly to incomplete full-matrix cells. Smoke outcomes are not mixed into headline matrices
unless their exact model, task, harness, and protocol identity match the frozen full run.

## Phase 2 freeze

Phase 2 starts only after all real matrices, splits, prices, and Phase 1 reports are immutable.
For each benchmark, WMO builds a model using only training-side real traces and Azure GPT-5.5 as
world-model inference. It replays the identical current roster, task IDs, attempt count, and router
grid into a separate matrix.

Comparison levels are:

1. Cell: agreement, absolute error, false positive, false negative, rank correlation when defined,
   and calibration band.
2. Model: Kendall and Spearman ordering, best-single agreement, quality ordering, cost ordering.
3. Route: selected-model, guard pass, route-away, and predicted versus realized delta agreement.
4. Decision: promote or reject, selected dial point, and realized quality and monetary consequence
   of disagreement.

Real and simulated rows are never pooled.

## Spend and durability

All historical exploratory calls made before this final goal are entered as prior spend. The
ledger counts successful, failed, incomplete, embedding, Azure, Anthropic, OpenAI, E2B, and
environment work. A launch is rejected when recorded spend plus a conservative remaining-cell
projection reaches USD 20,000.

Raw artifacts live under
`/Users/admin/Documents/experientiallabs/data/router-repro-20260728/full/` and remain out of Git.
Stable protocol, one-off runners, small manifests, hashes, summaries, and reports live under
`.agents/`. Every cell is persisted atomically before the next work unit. Long runs use tmux and
tee logs. A launch is accepted only after the completed-cell ledger advances on two polls.

## Frozen commands

The generated freeze summary records expanded non-secret arguments and SHA-256 identities. The
top-level commands are:

```bash
uv run python .agents/scripts/router_real_freeze.py --artifact-root \
  /Users/admin/Documents/experientiallabs/data/router-repro-20260728/full
uv run python .agents/scripts/router_real_smoke.py --resume
uv run python .agents/scripts/router_real_matrix.py --phase real --resume
uv run python .agents/scripts/router_real_analyze.py --phase real
uv run python .agents/scripts/router_real_world_models.py --resume
uv run python .agents/scripts/router_real_matrix.py --phase simulated --resume
uv run python .agents/scripts/router_real_analyze.py --phase simulated
uv run python .agents/scripts/router_real_compare.py
```

Commands whose runners do not yet exist are frozen interface contracts. Implementing them may fix
bugs but may not change the protocol, roster, tasks, policy grid, retry rule, or success gate.

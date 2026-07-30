# Coding model router experiment

Status: candidate only. The offline frontier clears the aggregate target, but fresh official-verifier promotion is still pending.

## Frozen candidate

The current candidate is a static reasoning-effort route:

- primary: `gpt-5.6-sol` with OpenAI Responses `reasoning.effort=xhigh`
- fallback: `gpt-5.5` with OpenAI Responses `reasoning.effort=xhigh`
- reference baseline: `gpt-5.5` at xhigh effort, matching the existing router baseline
- serving artifact: `.agents/policies/coding_router_small_agent_20260730.json`
- isolated source commit: `a734885b6a27224218ee73f1886ee44bb0ea697`

Reasoning effort is represented in `PoolEntry` and forwarded into `ProviderConfig`, so the selected effort is part of the serving contract rather than an undocumented launch flag.

## Fast proxy

The proxy contains 12 deterministic DeepSWE 1.1 tasks selected without using SWE-bench labels. Against the available external model snapshot, its rank correlation is:

| Comparison | Spearman rho | p-value |
| --- | ---: | ---: |
| proxy to DeepSWE | 0.9632 | 4.98e-7 |
| proxy to SWE-bench snapshot | 0.8441 | 5.54e-4 |

The full 12-arm DeepSWE to SWE-bench snapshot correlation is 0.7832 with p=0.00259. The snapshot is treated as an analysis input, not as a current official leaderboard.

## Reasoning-effort frontier

The report uses 113 DeepSWE tasks, task-level means, and five deterministic 70/30 task splits. It uses the available historical/shared ledger and is explicitly not a fresh execution.

| Arm | Mean quality | Mean cost per task |
| --- | ---: | ---: |
| GPT-5.5 xhigh reference | 0.6704 | $7.226 |
| GPT-5.6 Sol xhigh candidate | 0.7080 | $4.708 |

The candidate retains 105.61% of reference quality and saves 34.85% on the full ledger. Across the five held-out splits, mean quality retention is 102.40% and mean cost savings are 31.92%. The weakest held-out quality ratio is 98.98%, and the weakest held-out savings is 29.94%. The paired task-level quality-delta bootstrap 95% interval is -0.018 to 0.093, so fresh validation is still required.

The more expensive Sol max setting does not preserve the required cost band. This xhigh operating point is the main observed gain from tuning reasoning effort against the existing GPT-5.5 xhigh baseline.

## Transfer dataset

The router was also trained on 1,960 rows from `nvidia/Open-SWE-Traces`, config `openhands`, split `minimax_m25`, using prompt TF-IDF, language, trajectory steps, tool calls, and prompt length. Held-out accuracy was 0.5944 and ROC-AUC was 0.6483. This transfer signal was retained for analysis and rejected as the primary router because it did not improve held-out DeepSWE quality reliably.

## Fresh execution status

Fresh WMO official-verifier probes use unique experiment artifacts and the OpenAI and Anthropic credentials from the repo environment file without persisting secret values. Earlier GPT-5.6 Sol high probes returned zero whole-task rewards and strong partial scores, showing that the current Pi harness does not reliably cross the explicit submit boundary. The matched Sol xhigh and GPT-5.5 xhigh cells both stalled during E2B startup before producing verifier artifacts; their experiment-owned tmux processes were stopped with raw metadata preserved. The xhigh frontier is therefore promoted as an offline benchmark candidate only, pending a verifier-compatible rerun.

Promotion requires fresh matched verifier results, quality retention of at least 95%, and cost savings of at least 30%. Until then this document records a tuned candidate, not a completed router result.

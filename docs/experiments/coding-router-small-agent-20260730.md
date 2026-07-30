# Coding model router experiment

Status: candidate only. The offline frontier clears the aggregate target, but fresh official-verifier promotion is still pending.

## Frozen candidate

The current candidate is a static reasoning-effort route:

- primary: `gpt-5.6-sol` with OpenAI Responses `reasoning.effort=high`
- fallback: `claude-opus-5`
- reference baseline: `claude-opus-5` at the provider's high-effort operating point
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
| Claude Opus 5 high reference | 0.7286 | $6.088 |
| GPT-5.6 Sol high candidate | 0.6932 | $3.467 |

The candidate retains 95.14% of reference quality and saves 43.05% on the full ledger. Across the five held-out splits, mean quality retention is 97.62% and mean cost savings are 38.20%. The weakest held-out quality ratio is 90.98%, so the candidate is not promoted under a requirement that every split independently retain 95% quality. The paired task-level quality-delta bootstrap 95% interval is -0.114 to 0.041, which also warrants fresh validation.

The more expensive Sol xhigh and max settings do not preserve the required cost band. This is the main observed gain from tuning reasoning effort.

## Transfer dataset

The router was also trained on 1,960 rows from `nvidia/Open-SWE-Traces`, config `openhands`, split `minimax_m25`, using prompt TF-IDF, language, trajectory steps, tool calls, and prompt length. Held-out accuracy was 0.5944 and ROC-AUC was 0.6483. This transfer signal was retained for analysis and rejected as the primary router because it did not improve held-out DeepSWE quality reliably.

## Fresh execution status

Fresh WMO official-verifier probes use unique experiment artifacts and the OpenAI and Anthropic credentials from the repo environment file without persisting secret values. GPT-5.6 Sol high returned zero whole-task rewards in all completed fresh probes. The 60-turn probe reached 91.76% partial score, while the 120-turn probe reached 89.56% partial score, with no infrastructure error. This indicates the current Pi harness reaches substantial partial progress but does not cross the explicit submit boundary. The matched Opus 5 probe stalled before producing a verifier artifact and its experiment-owned tmux process was stopped with its raw metadata preserved.

Promotion requires fresh matched verifier results, quality retention of at least 95%, and cost savings of at least 30%. Until then this document records a tuned candidate, not a completed router result.

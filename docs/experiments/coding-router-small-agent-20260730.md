# DeepSWE coding-model router

Status: the profile router clears the evaluation gate against Luna @ max on grouped DeepSWE cross-validation. The result is a frozen-matrix replay, not a fresh set of model calls.

## What was trained and what was evaluated

The data contained 113 DeepSWE 1.1 tasks from 92 repositories. Each task had measured results for 13 model-and-reasoning-effort arms. The score was the graded fraction `f2p_passed / f2p_total`, with measured `cost_usd` for cost.

For each of five seeds:

- 70% of repositories were used to fit the router.
- 30% of repositories were held out for evaluation.
- No repository appeared in both sets.

The router used only the length of the task description. It split the training tasks into three length ranges, then selected the cheapest arm whose training quality was within 0.02 points of Luna @ max.

## Result versus Luna @ max for every task

| Method | Quality | Cost per task | Quality vs Luna @ max | Cost saving |
| --- | ---: | ---: | ---: | ---: |
| Luna @ max for every task | 0.958 | $3.00 | 100% | 0% |
| Profile router | 0.932 | $1.57 | 97.3% average, 95.1% worst split | 47.9% average, 30.4% worst split |

The router therefore saves about $1.43 per task while retaining at least 95% of Luna @ max quality on every held-out split.

## Frozen full-data policy

The full-data policy is saved at `.agents/policies/coding_router_deepswe_profile_20260730.json`:

| Task description length | Selected arm |
| --- | --- |
| 0 to 1,675 characters | Claude Opus 5 low |
| 1,676 to 2,608 characters | GPT-5.6 Luna xhigh |
| More than 2,608 characters | GPT-5.6 Luna xhigh |

The policy keeps Luna @ max as its pinned guard and fallback. The held-out evaluation is the meaningful result; the full-data policy is the artifact to serve after an external fresh validation.

## Turn-level cache and prefill behavior

The policy is now cache-aware when served live. It may reconsider the model on every turn. When a conversation has a known incumbent, the router compares the incumbent's warm cached-prefix prefill cost with the candidate's cold full-prefix prefill cost:

- It allows a cold prefill up to 4x the warm incumbent prefill when the profile points to a different arm.
- Otherwise it keeps the warm incumbent.
- The decision log records the cache credit and whether the switch passed or was reverted.

This is deliberately conservative. The current profile artifact has task-level quality evidence, but not turn-level quality evidence, so cache economics control switching while quality remains protected by the fitted task profile. The 47.9% cost result above is task-level; turn-level savings require a conversation trace evaluation.

## Remote-compute and reproducibility notes

The fitting sweep ran in an E2B sandbox. The laptop was used only to orchestrate the job, run small checks, and store artifacts. The report is at `.agents/reports/deepswe_profile_router_20260730.json`.

This is not a claim that the models were freshly rerun. It evaluates routing choices against the frozen 13-arm DeepSWE measurement matrix, so the next promotion step is a fresh matched DeepSWE execution using the repository's OpenAI and Anthropic credentials.

# ASSERT-KTH external expert-pool oracle gate

Date: 2026-07-31

Status: preregistered, outcomes unopened

## Question

Does a public external coding-agent matrix contain enough repeatable task-by-arm interaction to
justify another latency-neutral router experiment?

This is an oracle screen, not a router fit. It uses no DeepSWE reward, cost, verifier output, or
trajectory. DeepSWE task metadata is used only to remove exact task-id and normalized-prompt
overlap.

## Source

- Dataset: `ASSERT-KTH/agentic-evals-artifacts`
- Dataset commit: `5db0c4b69382d160a313d7ceaded915398c63e13`
- Benchmark: SWE-bench Verified
- Task metadata: `princeton-nlp/SWE-bench_Verified`
- Task metadata commit: `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`
- Arms: two scaffolds, three models, and two temperatures
- Independent attempts: ten per arm

Only the approximately 50 KB official SWE-bench result JSON from each run may be downloaded for
this gate. Full trajectories and detailed predictions are excluded.

## Frozen arm roster

1. `nano-agent-Qwen_Qwen3-32B-temp0`
2. `nano-agent-Qwen_Qwen3-32B`
3. `nano-agent-agentica-org_DeepSWE-Preview`
4. `nano-agent-agentica-org_DeepSWE-Preview__temp0`
5. `nano-agent-mistral_devstral-2512`
6. `nano-agent-mistral_devstral-2512__temp0`
7. `r2e-gym-Qwen_Qwen3-32B`
8. `r2e-gym-Qwen_Qwen3-32B__temp0`
9. `r2e-gym-agentica-org__DeepSWE-preview`
10. `r2e-gym-agentica-org__DeepSWE-preview__temp0`
11. `r2e-gym-mistral_devstral-2512`
12. `r2e-gym-mistral_devstral-2512__temp0`

The `DeepSWE-Preview` string is a public model identity in the external source. It does not mean
that DeepSWE v1.1 target outcomes are used.

## Held-out oracle

For each of 400 deterministic attempt splits:

1. Use five attempts per arm to select the best arm for each task.
2. Use the other five attempts to score that choice.
3. Select the best static arm using only the five fit attempts.
4. Score the static arm on the same five held-out attempts.
5. Measure held-out oracle reward minus held-out static reward.

Tie breaking uses fit data only: global fit reward, then frozen arm order. The uncertainty
distribution combines attempt splits with repository bootstrap resampling. Repositories, not
individual tasks, are the sampling unit.

## Frozen gate

Proceed to cost extraction and router fitting only when every condition passes:

1. At least 250 tasks remain after exact task-id and normalized-prompt overlap removal.
2. At least eight arms have ten complete attempts on the same retained task cohort.
3. Mean held-out oracle headroom is at least 0.10 absolute reward.
4. The combined repository-and-attempt 95 percent interval lower bound exceeds 0.05.
5. Every retained arm has exactly one gradeable binary outcome per task and attempt. Missing,
   incomplete, empty-patch, and agent-error submissions score zero. Infrastructure omissions are
   not silently dropped.

The threshold is intentionally stronger than merely excluding zero. It seeks the ACRouter-shaped
regime that the saturated DeepSWE arm pool lacked.

## Stop rule

If the gate fails, do not inspect trajectory features, fit a router, extract per-task costs, or
open another DeepSWE replay. Preserve the report as a negative result and search for a different
external expert pool.


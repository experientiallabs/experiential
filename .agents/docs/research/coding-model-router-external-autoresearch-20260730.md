# External coding trace router autoresearch

Date: 2026-07-30

Status: static-frontier checkpoint reproduced in native WMO, task-routing gain not established

## Objective

Improve DeepSWE v1.1 cost efficiency without fitting any parameter, feature transform, or
threshold on DeepSWE outcomes. The routing action is model plus reasoning effort. The deployed
router must run before inference without another language model call or feedback loop.

## External fit protocol

The first frozen fit used 4,484 deduplicated tasks from four sources:

| Source | Raw tasks | Fit tasks | Repository groups | Weak reward | Strong reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nebius SWE Agent 8B and 70B | 527 | 527 | 375 | 0.1133 | 0.1368 |
| R2E Gym GPT-5 Codex and Kimi 2.5 | 3,393 | 3,393 | 10 | 0.4960 | 0.7135 |
| SWE-bench Verified GPT-5.2 effort IRT | 500 | 500 | 12 | 0.6900 | 0.7180 |
| CodeRouterBench OOD176 | 176 | 64 | 22 | 0.3295 | 0.6420 |

The fitter used repository-grouped five-fold cross-validation, source-balanced operating-point
constraints, a task-blind control, a within-source shuffled-label control, and leave-source-out
checks for the two main Ridge candidates. It froze one candidate and the external quality floors
0.95, 0.97, and 0.99 before opening the target artifact.

The selected scorer was `word128-ridge-heads-a1`. Its external out-of-fold uplift Spearman was
0.0657. At the external 0.95 quality floor it sent 49.83 percent of source-balanced traffic to
the strong arm, retained 96.20 percent mean quality, and retained at least 90.02 percent on every
source.

## Frozen DeepSWE checkpoint

The target replay used 110 fully scored tasks across 88 repositories and graded
`f2p_passed / f2p_total` reward. Static comparisons span 41 model and reasoning-effort arms.
Repository bootstrap resamples repositories, not individual tasks.

The least-cost preregistered point that passed the promotion rule was:

| Field | Result |
| --- | ---: |
| Ladder | GPT-5.6 Luna xhigh to max |
| External operating point | 0.97 |
| Target traffic | 26 xhigh, 84 max |
| Router quality | 0.9392186960 |
| Best static quality | 0.9543283155 |
| Quality retention | 98.4167 percent |
| Router cost | USD 316.1660 |
| Best static cost | USD 679.8643 |
| Cost savings | 53.4957 percent |
| Paired quality delta 95 percent CI | [-0.0437750, 0.0091836] |
| Allowed quality delta | -0.0477164 |
| Static dominated | no |
| Promotion | pass |

The external 0.99 point also passed at 99.0667 percent quality retention and 50.1149 percent
cost savings. The selection rule chose the cheaper passing point.

This is deployment calibration, not untouched confirmation. The ladder came from previously
known DeepSWE aggregate arm results, even though target task labels did not enter the fit or
thresholds.

## Weighting audit correction

The first fit exposed a data-combination bug after the target replay. CodeRouterBench has 176
unique prompts internally, but 112 are exact normalized-text duplicates of tasks already loaded
from earlier sources. Keeping 64 rows is the correct cross-source deduplication result. The bug
was that weights were assigned before deduplication, so CodeRouterBench received 64/176 of the
intended source weight while the other sources each received their full weight.

This is not target leakage, but it makes the recorded `equal_total_weight_per_source` claim false
for the first checkpoint. Commit `20da7f5e` changes deduplication to exact normalized text,
retains distinct prompt variants with collision-safe ids, and computes weights after
deduplication. Commit `caf60b81` records per-source weight totals in the fit artifact. The
corrected artifact reports 1,121 total weight for each of the four retained sources.

## Native serving policy

The original selected joblib is 48,671,892 bytes because it carries a learned vocabulary and SVD
transform. It is a research artifact, not a serving requirement.

The native policy uses WMO's deterministic signed character-trigram hashing embedder plus two
Ridge potential-outcome heads. The artifact stores only plain JSON weights, biases, the frozen
threshold, and the two model plus reasoning-effort pool entries. Serve time performs one local
embedding and two dot products. It makes no network call and loads no pickle.

`RoutingPolicy(kind="linear")`, shared offline replay, save and load, validation, sticky routing,
Pareto reporting, and the OpenAI-compatible chat endpoint are covered by tests. The adaptive
native search uses only external outcomes, but the family was chosen after the first DeepSWE
result and must not be reported as untouched target confirmation.

The corrected native external fit selected `hash512-ridge-heads-a1`. Source weights are equal
after deduplication. Its external out-of-fold uplift Spearman is -0.0157, which does not support a
general task-text uplift signal.

The external 0.97 point still passes the original static-frontier promotion gate:

| Field | Corrected native result |
| --- | ---: |
| Target traffic | 40 xhigh, 70 max |
| Router quality | 0.9382796991 |
| Quality retention vs best static | 98.3183 percent |
| Router cost | USD 295.3985 |
| Cost savings vs best static | 56.5504 percent |
| Paired quality delta 95 percent CI | [-0.0442314, 0.0067431] |
| Static dominated | no |
| Original promotion gate | pass |

That gate is insufficient. A 10,000-sample task-blind control randomly assigns the same 70 tasks
to max effort. Its expected quality is 0.9392274 and expected cost is USD 283.9068. The router is
0.0009477 lower quality and USD 11.4917 more expensive than that mean. Its quality is at the
39.58th percentile and its cost is at the 89.71st percentile of matched random assignments.

Conclusion: the native artifact works and reproduces the fitted scorer exactly, but the measured
benefit is effort mixing, not learned task routing. It must not be promoted as a routing
algorithm. The next algorithm must beat a matched task-blind mixture, not only discrete static
arms.

The same control rejects the original word plus SVD router. At its selected external 0.97 point,
the router sends 84 of 110 tasks to max effort. A 10,000-sample task-blind control with the same
traffic has expected quality 0.9414687 and expected cost USD 306.4853. The router is 0.0022500
lower quality and USD 9.6808 more expensive than the random mean. Its quality is at the 23.26th
percentile and its cost is at the 88.92nd percentile. The 0.99 point is statistically
indistinguishable from the matched mixture. Both tested lexical feature families therefore fail
to establish task-selection value.

## Open-SWE external transfer

The next source uses
[`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) outcomes joined
by external task id to issue text from
[`nebius/SWE-rebench-V2`](https://huggingface.co/datasets/nebius/SWE-rebench-V2). A projected
parquet scan reads identity and outcome columns instead of downloading the 18.3 GB trajectory
payload. The compact paired dataset stays in E2B.

Within each agent scaffold, the preparation rule selects the two model modes with the largest
external mean-reward gap. It selected OpenHands with Qwen3.5-122B as weak and MiniMax-M2.5
thinking as strong. After joining text there are 14,504 paired tasks across 2,251 repositories
and nine languages. Their mean task rewards are 0.3230 and 0.3957. The fit receives no DeepSWE
path.

The first family uses 27 deterministic issue-shape features. Its IRT variants predict task
easiness, calibrate weak and strong ability offsets, and assign stronger effort to the
middle-difficulty band. The IRT hypothesis failed externally: every observed IRT variant had
negative out-of-fold uplift Spearman. The preregistered family instead selected a two-head Ridge
baseline with 0.0529 out-of-fold uplift Spearman.

The selected external 0.97 point is directionally better than its matched task-blind DeepSWE
control but is not significant:

| Field | Structural two-head result |
| --- | ---: |
| Target traffic | 26 xhigh, 84 max |
| Router quality | 0.9430970943 |
| Router cost | USD 304.1929 |
| Matched task-blind expected quality | 0.9414687144 |
| Quality delta vs task-blind mean | +0.0016283799 |
| Matched task-blind expected cost | USD 306.4853 |
| Cost delta vs task-blind mean | -USD 2.2924 |
| Router quality percentile | 68.35 percent |
| Router cost percentile | 37.05 percent |

The external 0.99 point reaches the 96.71st quality percentile but costs USD 5.9150 more than the
matched task-blind mean. This is weak task-selection evidence, not a promotable result.

## Reproducibility anchors

First external fit commit: `bbbaa609aa7f8b9e6a35aab311920ad11ef17266`

Frozen target grid report commit: `10cdd7c7`

Native policy commit: `3942ca6a`

Corrected source weighting commit: `20da7f5e`

Task-blind control commit: `962bf990`

`xhigh` model-pool contract commit: `1eef58e6`

First selected joblib SHA256:
`4eccda3b30b5f134691159cc003813e59d0a7dfe56e841742b3901988d599a96`

Frozen candidates SHA256:
`c8f3b96d62d82bf905b5446acf9b851485449f2283ecd3ec4305c61fba1f5fd8`

Frozen DeepSWE grid SHA256:
`2f6a22a2845bb0fec8f66233b7f26fd5b20f512950f6ef391f6bf216881a3cb2`

E2B template: `deepswe-router-cpu8gb-v1`

E2B sandbox: `idqwkvv60h7weldgl08p8`

Corrected native heads SHA256:
`88cbea68922457343781d2ba19c7404bf456e8b4686d0f1e8ea41beb170d4087`

WMO native policy SHA256:
`95826f5f31e3d2100208e734a03f96ff96fd8c9ea8a4517c424bde5cbc09e72f`

Serving parity report SHA256:
`b156a437e9ee70617c00d6ae62d4ce9511c91de9ce3c1993e619f82b69f34ba4`

Matched task-blind control report SHA256:
`7aaa55de211b5f1deb63c9437acf5440344a39a931fcb295b6018d5e285d86cc`

Original word-router task-blind control report SHA256:
`5dc64e22fbab0ebf10e8b62f5cbb81b90251ff4cc219b6992b2af119b402492c`

Open-SWE source preparation commit: `da8b1f8e`

Structural IRT family commit: `bd7f8f82`

Open-SWE compact source SHA256:
`179d9801507b514ec30eb279cf44235ee4b6634bf38b06cee741a1018c391d55`

Structural frozen candidates SHA256:
`4ca5cf101a358551fa077124a9f874854625ae9c00b5ef2e757546ac160f4efa`

Structural DeepSWE report SHA256:
`47ab1efc7e4556f6fd31b4af858073132a80fd569f5a65b2e11b31e094de3b4d`

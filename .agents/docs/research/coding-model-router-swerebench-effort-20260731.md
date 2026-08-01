# SWE-rebench V2 reasoning-effort routing protocol

Status: frozen before provider execution on 2026-07-31; DeepSWE outcomes remain sealed.

## Objective

Learn a latency-neutral `gpt-5.6-luna` reasoning-effort router from verified
repository tasks that are disjoint from DeepSWE, confirm exactly one frozen
route on untouched repositories, and permit one DeepSWE transfer only if every
external promotion gate passes.

The router may use only deterministic label-free request metadata available on
both corpora. It may not call a model, retrieve a demonstration, inspect a test,
or scan a repository at inference time. No fitted foundation model is retained.

## Source and target isolation

The source is
`PrimeIntellect/SWE-rebench-V2-Filtered-Verified` at revision
`03cc767ee33126b7fc7890ad57047e9dd6914cca`. Its single Parquet shard has
SHA-256 `7416e352008b35480c82610cf4f5edf160dd269ed6bf382a22bd4c17daed24b9`.
Selection read only task identity, prompt, repository, language, base commit,
creation time, and image identity. It did not read gold patches, test patches,
test lists, verifier configuration, or LLM-authored metadata.

The label-free DeepSWE task index has SHA-256
`b0d25ec0e566c0391e4385a63343b92d5371b67f052e1c9062c9d226d9d18dd1`.
The complete 113-task prompt view has SHA-256
`35ad33855f63f147b1861b58b59ad635f8860677b5d0d5e902c421029d78637b`.
No target reward or cost field was read.

A deterministic SHA-256 ordering with seed `20260731` froze two 200-task
cohorts. Each cohort contains 60 Go, 10 JavaScript, 60 Python, 10 Rust, and 60
TypeScript tasks, with at most three tasks per repository. Development and
confirmation each span 103 repositories. They have zero repository overlap
with one another, zero repository overlap with DeepSWE, and zero normalized
exact-prompt overlap with DeepSWE.

| Artifact | SHA-256 |
| --- | --- |
| Development tasks | `7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1` |
| Confirmation tasks | `9798dd1e58be0d13331d097307670dc3fc3760ad211da20e6367666523f080a7` |
| Cohort manifest | `78452c079f399b7de9cf73720f516b93c0b988fb436e328325fd79d4d04c6eb2` |

Confirmation tasks and outcomes remain unavailable to fitting until one
development candidate and every confirmation gate are content-addressed.

## Execution pins

- Model: `gpt-5.6-luna`
- Reasoning efforts: `low`, `medium`, `high`, `xhigh`, `max`
- Attempts per task and effort: two
- Development cells: 2,000
- Confirmation cells if authorized: 2,000
- Temperature: 1.0
- Per-response output limit: 32,768 tokens
- Per-agent limits: 20 model turns, 131,072 total output tokens, 900 seconds
- Setup and official scoring limit: 900 seconds each
- Harness: verifiers `mini_swe_agent` version 2.4.5
- Verifiers commit: `f6e420b9908ae14d625f079881f13c15011ee1c9`
- Official taskset commit: `a90fbd708de9ab18f85b5ffc3a0bdc60825dcc84`
- Base runtime: task-specific Docker image inside model-free E2B template
  `deepswe-router-docker-v1`, template ID `3v4ie6miz6uhlhhfeyea`, build ID
  `95fa38ac-01fa-45b5-841d-72e17fce0819`
- Final runtime: model-free E2B template `deepswe-router-responses-v2`,
  template ID `j1a2bxbpllu3rp84b4qj`, build ID
  `e971c040-95bd-45c1-89ee-fb597bf75671`; it derives from the base and bakes
  in only the two content-checked source adapters described below
- E2B account cap: 1,000 sandboxes

The filtered dataset rewrites each upstream
`docker.io/swerebenchv2/<name>:<tag>` image to the Prime-only alias
`prime/primeintellect/<name>:<tag>`. E2B's Docker runtime cannot resolve that
alias. Before loading tasks, the runtime verifies the pinned official taskset
source hash
`b7dab7d2b263d6d296cc9e2e4b9b4597cc3fbba040d3036a139cf0fb4432e730`
and applies
`.agents/scripts/coding_model_router_swerebench_docker_adapter.py`. The adapter
reverses exactly the documented registry rewrite and changes no prompt, task,
test, verifier, resource, or reward field. Its preimage hash, postimage hash,
and mapping report must persist with every run.

The official `swerebench_v2_v1` taskset restores test files, applies the hidden
test patch only at scoring time, runs the pinned test command, and returns one
only when every fail-to-pass and pass-to-pass test passes. Empty patches,
agent errors, provider truncations after model execution, and unresolved tasks
are gradeable zero-reward model outcomes. Failures before model execution or
before an official verifier result are infrastructure failures and may be
resumed without changing the scientific attempt identity.

The runtime must attest the requested model and reasoning effort on every
provider turn. Exact usage telemetry is retained when available. Otherwise the
ledger uses the existing trace-derived estimator and labels that provenance.
Frozen list prices are USD 1.00 per million input tokens, USD 0.10 per million
cached input tokens, and USD 6.00 per million output tokens.

## Infrastructure repair record

The first smoke invocation reached zero provider turns in all four cells and
stopped with `SandboxError` because Docker treated the Prime aliases as Docker
Hub names. These are infrastructure failures, not model outcomes. The raw
failed artifacts remain under
`/private/tmp/coding-router-swerebench-v40-smoke`; no cell was replaced and no
scientific attempt number was consumed.

A zero-provider E2B probe on 2026-07-31 verified the adapter against the pinned
taskset source. It pulled both exact frozen images from Docker Hub:

- `0xs34n-starknet.js:538-72d73f6`, image ID
  `sha256:bab8a1ea7e8dd5755faed8f0775f94f94f1cea0f8812d3f88d425526f61461cc`
- `acloudguru-serverless-plugin-aws-alerts:13-3d02390`, image ID
  `sha256:324c193d3ffd988872c40b5a39866b28f028ae04059b285f71e792af516d7d9d`

The adapter output hash was
`a2790c3f296a28f40eb8732d68c091cc7b9899e08916aedec6b2b53a644f7b3e`.
The exact probe sandbox `ia52cg7dc80uxregp3qr6` was terminated after artifact
sync. Resume must use the same four task, effort, and attempt identities.

The repaired smoke then exposed a pinned verifiers v1 client incompatibility
before inference: its Chat Completions proxy serialized the configured output
limit as `max_tokens`, which `gpt-5.6-luna` rejects in favor of
`max_completion_tokens`. All requests returned HTTP 400 with no inference and
consumed no scientific attempt. The exact owned eval process was terminated
and its sandbox retained for recovery. Before resuming, the runtime verifies
the pinned `verifiers/v1/dialects/chat.py` source hash
`47fa2daa2e4dd2c9c1d5054a21896835e2c747d91cece988d3a1d88358abfcbc`
and applies `.agents/scripts/coding_model_router_verifiers_luna_adapter.py`.
That adapter preserves the same 32,768-token limit and changes only its wire
field name for the frozen model.

The next pre-inference request established that Luna does not support function
tools plus reasoning effort on Chat Completions. Mini-swe-agent 2.4.5 already
ships a `litellm_response` model class, and the pinned verifiers proxy already
ships a native Responses dialect that maps `max_tokens` to
`max_output_tokens`, maps reasoning effort to `reasoning.effort`, and preserves
encrypted reasoning state. The final runtime therefore verifies the pinned
mini-swe-agent harness source hash
`5c898dbf5fb3eb350290e193f90341ea80da705e2ea506cc5f37450c86314a78` and
applies `.agents/scripts/coding_model_router_verifiers_responses_adapter.py`.
It changes only the harness model class from `litellm` to
`litellm_response`; the agent version, prompt, tools, turn limits, runtime,
model, effort, token limit, and verifier remain frozen. The prior Chat
Completions token adapter is retained only as evidence for the failed transport
probe and is not required by the final Responses runtime.

The same four scientific cells then completed through the Responses runtime in
owned sandbox `izarp6idyx2t2r1m1zef6`, which was terminated after raw artifact
sync. Both efforts solved
`acloudguru__serverless-plugin-aws-alerts-13` and scored zero on
`0xs34n__starknet.js-538`; this runtime smoke is not evidence of a routing
effect. The `xhigh` and `max` archives have SHA-256 values
`bf1d576d25f1b56ae3a9484db5d5599576519a218aec3073db29272345f4015b`
and
`c449dc999a4d604546c358affcf5e1cba1865aba8ca312789b92b5eb27bb4e6a`.
The strict audit verified all four official rewards, every call's model,
effort, Responses endpoint, output limit, and token usage, the official scoring
timings, patch hashes, archive hashes, unchanged attempt identities, and
sandbox termination. It counted 62 provider calls, 152,889 uncached input
tokens, 1,343,252 cached input tokens, 25,106 output tokens, and 10,646
reasoning tokens included within output. Frozen list prices give a trace-derived
estimate of USD 0.437850. The canonical audit report is
`/private/tmp/coding-router-swerebench-v40-smoke-resume-1/smoke-report.json`,
SHA-256
`ee76a57040cbe7aaef692d2fc3f3df66d7a556cbf6dda74119e0802cb4230e13`.
It explicitly corrects the pre-audit state file's stale zero provider-call
count to the audited value 62.

## Smoke and matrix launch

Run exactly four cells first: the first two development tasks by frozen task
order, at `xhigh` and `max`, attempt zero. The smoke must prove:

1. Docker starts inside E2B and pulls both task images;
2. the pinned taskset and mini-swe-agent harness load;
3. all provider turns attest `gpt-5.6-luna` and the requested effort;
4. each cell reaches the official verifier or a gradeable post-execution zero;
5. traces, rewards, patches, usage provenance, and resume state persist;
6. the exact E2B smoke sandboxes are terminated after artifact verification.

If the smoke exposes an infrastructure defect, repair and resume only these
four cells. Do not launch a replacement smoke. If the smoke is valid, run all
2,000 development cells with bounded E2B concurrency. Reuse completed cells by
content hash and never rerun a gradeable model outcome.

The development execution uses 25 E2B workers. Each worker owns one task,
pulls its single frozen image once, and executes two attempts per effort. A
five-way cyclic rotation of the frozen effort tuple by task index balances
effort order across the 200 tasks. The first two tasks' `xhigh` and `max`
attempt-zero cells are reused from the valid smoke by archive hash, so the
development launch creates 1,996 new cells and does not rerun those four
gradeable outcomes. Each effort archive and compact audit report is synced and
hashed before its task state advances. A worker failure leaves completed
efforts immutable and resumes only missing scientific cells.

The development validator initially required token usage on every provider
attempt. Three otherwise gradeable cells included one recorded OpenAI 503
attempt with null usage, followed by additional successful model turns and a
completed official zero reward. The failure was accounting-only: the exact
owned sandboxes and raw traces were preserved, a content-equivalent validator
accepted null usage only for an explicit HTTP 429 through 599 provider error,
and the three effort artifacts were recovered without rerunning a scientific
cell. Such attempts are counted separately from inference calls and assigned
zero usage cost because the provider returned no inference usage. All live
workers received the same validator repair before their next validation.

## Target-compatible candidate families

Development may compare the following lightweight families. All use only the
problem statement, canonical repository string, language, and deterministic
prompt-shape features. Base-commit identity may be retained for auditing but
not as a fitted feature.

1. Static effort arms and task-blind mixtures.
2. Signed character 3-to-5-gram hashing with Ridge reward-delta heads relative
   to high effort. Hash dimensions are 512, 2,048, and 8,192; Ridge alphas are
   1, 10, and 100.
3. Monotone ordinal reward heads on the same representations. Predicted reward
   is isotonic across effort before cost-aware selection.
4. Pairwise uplift heads for adjacent effort steps with repository-grouped
   cross-fitting and inverse-variance shrinkage.
5. A two-parameter item-response model whose task difficulty and discrimination
   are predicted from the same representations.
6. WMO guarded kNN using cached deterministic vectors, `k` in 8, 16, 32, and
   64, asymmetric guard, `knn_z` in 0, 0.5, 1, 1.645, and 2, and `pick_lam` in
   0, 0.01, 0.02, and 0.03. No absolute similarity floor is allowed.

Within five development outer folds grouped by repository, fit-only selection
chooses the least costly point within 0.5 reward points of the fold's strongest
eligible quality point, then breaks ties by higher reward, lower route latency,
and simpler family order above. A candidate must beat its matched task-blind
mixture in aggregate, retain at least 95 percent of the strongest fit-selected
static reward in every outer fold, and avoid static dominance. Development is
adaptive within these families, but it may freeze only one confirmation rule.
No confirmation result may change its features, hyperparameters, thresholds,
guard, arm roster, or tie breaks.

## Confirmation gates

The frozen route advances only if the untouched 200-task confirmation matrix
satisfies all of the following:

1. every task-effort cell is gradeable after taskwise removal of any genuinely
   missing infrastructure cell, and retained task coverage is at least 95
   percent;
2. route reward minus a matched task-blind mixture with identical effort counts
   is positive and its 10,000-resample repository-bootstrap 95 percent lower
   bound is above zero;
3. no single static effort has at least the route reward at no greater cost;
4. the same fitter with a fixed within-development-repository outcome
   permutation fails the primary matched-blind gate;
5. every development outer fold retains at least 95 percent of its strongest
   static reward and the frozen family retains that condition on confirmation;
6. the complete pre-inference route takes less than 5 ms per task on the E2B
   reference worker; and
7. the isolation audit still reports no target outcome access and no source to
   target repository or exact-prompt overlap.

A failed confirmation is final for the frozen route. Confirmation-dependent
tuning or another confirmation cohort is prohibited.

## Single sealed DeepSWE transfer

Only a fully passing external confirmation can authorize one target transfer.
Before target outcomes are opened, refit the exact frozen candidate on the
development outcomes only, write all 113 effort decisions against the frozen
label-free target view, and content-address the complete route file on a
no-internet E2B worker. The fitted numeric state remains ephemeral.

Then evaluate those frozen decisions exactly once against the existing
DeepSWE matrix with SHA-256
`2988742e48b1c9bfec8dc45d88af112c46c45367529d1936b709e4b4e549835f`.
Use graded fail-to-pass reward and measured trial cost, remove a task across all
five efforts if any required cell is missing, compare every static effort and a
matched task-blind mixture, and bootstrap router-minus-blind reward by
repository. There is no second target evaluation and no target-dependent
refit, threshold, representation, or arm change regardless of the result.

## Budget and stopping

Trace-derived experiment spend after the valid smoke is approximately USD
405.767850. The user monitors provider billing externally and authorized a USD
20,000 hard ceiling. Exact metering is not a launch gate. Preserve exact
telemetry where available, otherwise update a clearly labeled trace-derived
estimate after each completed tranche. Stop before any launch whose estimate
would take total experiment spend above USD 20,000.

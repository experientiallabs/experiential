# Graded external reasoning-effort matrix

Status: frozen before cohort preparation or paid execution on 2026-08-01. DeepSWE outcomes remain
sealed.

## Question

Can a larger external repository-task matrix scored by graded fail-to-pass test progress support a
practical reasoning-effort policy that binary SWE-rebench outcomes could not?

This experiment changes the two factors identified as load-bearing by the positive DeepSWE work:
reward shape and reasoning-effort coverage. The previous 194-task external matrix used binary
resolution. Binary scoring made the static arm gap much larger than the graded DeepSWE gap and
forced every quality-safe learned policy to route mostly to `sol-max`. The new matrix uses the
fraction of curated fail-to-pass tests passed, which is the target benchmark's optimization
metric, while keeping every task and outcome external to DeepSWE.

## Frozen source cohort

The source task pool is `nebius/SWE-rebench-V2`, local Parquet SHA-256
`0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad`. The starting
selection is the independently mined 1,000-task k-center cohort at
`/Users/admin/Documents/experientiallabs/rebench-mining/mined_1000.json`, SHA-256
`40f9a1b3ace2a592cbdfeac54b57db4f4638b3477df00693207ea45ab88f6caa`. Its selection
used public task metadata, code embeddings, repository and language caps, recency, public Qwen
trajectory difficulty, and gold-patch complexity proxies. DeepSWE task similarity was calculated
only after the ranked selection and did not affect selection order.

Before splitting, remove any source task whose repository or normalized prompt overlaps the
label-free 113-task DeepSWE index. Also remove duplicate identities, empty prompts, invalid Docker
images, and tasks with no fail-to-pass tests. Gold patches, test patches, and test identities may be
used by the external verifier but are forbidden as router inputs. At least 900 tasks must survive.

Assign complete repositories to development or confirmation with frozen seed 20260801 and an
approximately 70/30 task split. Repository overlap must be zero. The split, task identities,
pre-call text, images, verifier data, and all input hashes are frozen before provider calls.

## Frozen arm roster and execution

The six arms are:

- `gpt-5.6-luna` at low, medium, high, xhigh, and max reasoning effort;
- `gpt-5.6-sol` at max reasoning effort as the frontier guard candidate.

Each task-arm cell receives exactly one attempt with the pinned mini-swe-agent 2.4.5 harness,
20-turn limit, OpenAI Responses adapter, Docker task image, and verifiers commit
`f6e420b9908ae14d625f079881f13c15011ee1c9`. Each sandbox runs exactly one task-arm cell within
the workspace's one-hour lifetime, with up to 100 tasks advancing concurrently under the 1,000
sandbox E2B cap. Every completed cell is persisted immediately and the run is resumable.

The pinned SWE-rebench taskset source SHA-256 is
`a2790c3f296a28f40eb8732d68c091cc7b9899e08916aedec6b2b53a644f7b3e`. The only
scientific verifier change replaces binary all-tests resolution with
`passed FAIL_TO_PASS / total FAIL_TO_PASS`, using the same parser, normalization, hidden test patch,
test command, and scoring pass. The patch and patched source hash are recorded and validated in
every worker. The resulting patched taskset SHA-256 is
`cf920fb55d6704da9ba0b6fe7cf676fdac8ec1aeb719c3640f713fdbf7ad0cce`. Its two
non-scientific changes accept the source dataset's direct Docker image name and load one frozen
local task JSON, avoiding a Hugging Face download in every sandbox. No LLM judge or patch
similarity proxy is used.

The frozen cohort retains 993 of 1,000 tasks after excluding seven target-repository overlaps.
Development contains 673 tasks across 489 repositories and confirmation contains 320 tasks across
218 repositories. Their repository overlap and their repository and normalized-prompt overlap
with DeepSWE are all zero. The label-free development and confirmation manifest hashes are
`48d88436a083b66972c25cd7d9439fd149c95bcf9caded2bab7f3b6453aea3d5` and
`c9443c9956e496123f396ee793efbb3368312092c4dcbd4e5e10bb77bd814f0a`;
the private verifier manifest hash is
`bebfbf48f3d0b6f0fca6715c39dffb17c2bec44b52780ddfac7d812f0f3673f8`.

Agent failures after a gradeable patch are scored outcomes. Infrastructure failures before a
gradeable result may be retried once under the frozen policy. Any irrecoverable missing cell causes
whole-task exclusion across all six arms. At least 95 percent whole-task coverage is required.

## Frozen analysis

The confirmation outcomes remain unopened until a policy is selected using development only. The
fit-selected strongest static arm is the baseline. Report all six static arms, cheapest arm, full
oracle, pair oracles, and WMO's guarded kNN policy using only pre-call repository, language, prompt,
and cached coding-task embeddings.

Development selection uses repository-grouped five-fold cross-validation at seeds 11, 23, 37, 41,
and 59. Arms remain model by reasoning-effort identities. WMO kNN uses the existing paired guard,
asymmetric cost guard, minimum eight paired neighbors, small-sample standard-error floor, and cost
dial capped at 0.03. Candidate neighbor counts are 8, 16, 32, and 64; z values are 0, 0.5, 1,
1.645, and 2; and every fit-frontier arm is considered as a pinned guard. Selection minimizes cost
subject to every seed retaining at least 95 percent of the fit-selected static quality, saving at
least 40 percent, positive matched task-blind advantage, and no static dominance.

The selected point is evaluated once on the sealed repository-disjoint confirmation split. It
must retain at least 95 percent quality, save at least 40 percent, clear a repository-bootstrap
paired quality lower bound, avoid static dominance, and route in under 5 ms p95. If it passes, the
frozen policy receives exactly one DeepSWE v1.1 transfer with graded fail-to-pass reward. No target
repair, hyperparameter change, or rerun is allowed.

The paired uncertainty gate uses 10,000 repository-cluster bootstrap draws with seed 20260801. Each
draw resamples confirmation repositories with replacement and computes mean
`router reward - 0.95 * fit-selected-static reward`, retaining original within-repository task
clusters. Its 2.5th percentile must be nonnegative. This exact interval is frozen before any
confirmation outcome is collected.

The protocol's existing matched task-blind requirement is also applied mechanically on
confirmation. For each task, the control reward is the router's aggregate model traffic dotted
with that task's six arm rewards. A second 10,000-draw repository-cluster bootstrap with seed
20260802 estimates `router reward - identical-traffic task-blind reward`; its 2.5th percentile must
be strictly positive. The confirmation report includes all 15 pair oracles as well as the full
oracle. These checks were implemented while confirmation remained sealed and do not change the
development search or selected route.

## Compute, persistence, and spend

All cohort transformation, fitting, and analysis run on E2B or Azure. The Mac only orchestrates,
hashes, and stores bounded artifacts. No fitted model is persisted. Raw task, verifier, and trace
artifacts stay out of Git.

The user authorized a USD 20,000 total hard ceiling and monitors provider usage externally. Rough
cumulative spend before this campaign is USD 3,025.10805955. The prior measured six-arm cost on
194 external tasks projects this one-attempt matrix near USD 1,563 per 1,000 tasks, but this is only
a trace-derived planning estimate. The campaign stops before the total ceiling and preserves exact
token telemetry when available.

## Launch audit

The first controller launch made zero provider calls because E2B rejected six-hour sandbox
lifetimes before creating a worker. The corrected execution uses one cell per one-hour sandbox.
After paid work began, the local artifact validator incorrectly rejected an officially scored
no-change trace whose captured patch was null. The audit-only validator was updated in all active
experiment-owned sandboxes without changing any scientific evaluation, and a zero-provider watcher
keeps that correction in newly created workers from the already-running controller. Six cells
completed before their validator was updated and their remote traces were lost on worker
termination. Those six tasks are permanently excluded whole-task with zero reruns. The resulting
667 of 673 initial maximum development coverage was 99.1 percent.

The first watcher revision could write before the controller's startup validator write and then be
overwritten. It also raced one controller write, producing an invalid audit script. Three more
scientifically completed cells lost only their local audit artifacts: `prestashop__prestashop-27425`,
`neurodatawithoutborders__pynwb-439`, and `prestashop__prestashop-37692`. All three tasks are
permanently excluded whole-task with zero scientific reruns. The watcher now waits until the
controller persists the pulled Docker image identity, which occurs after the controller validator
write, before installing the corrected audit-only validator. The resulting 664 of 673 maximum
development coverage is 98.7 percent, above the frozen 95 percent requirement.

One later `luna-high` cell for `open-telemetry__opentelemetry-swift-763` completed its single
frozen agent attempt but produced no official graded reward. Its worker had already terminated, so
the trace and provider usage were irrecoverable. The task is permanently excluded across all six
arms with zero reruns under the frozen missing-cell rule. The resulting 663 of 673 maximum
development coverage is 98.5 percent, above the frozen requirement.

A later E2B HTTP/2 control-plane failure affected seven task workers. Five failed before their
scientific command started and remain eligible for the single frozen infrastructure retry.
`switchbacktech__compass-519` completed its `luna-max` command without a graded reward, and
`azuread__microsoft-authentication-library-for-python-315` started its `luna-xhigh` command before
the transport failed and its sandbox was terminated. Neither scientific cell can be recovered or
rerun. Both tasks are permanently excluded whole-task. The resulting 661 of 673 maximum
development coverage is 98.2 percent, above the frozen requirement.

The `sol-max` scientific command for `rustls__rustls-2022` later completed after three other arms
had produced valid artifacts, but its trace contained no official graded reward. The worker had
terminated and its provider usage was not recoverable. The task is permanently excluded across all
six arms with zero reruns. The maximum retained development cohort is now 660 of 673, or 98.1
percent, still above the frozen requirement.

The first `luna-max` command for `vmware__govmomi-3628` also completed successfully, but its
official trace contained no graded reward. The worker had terminated before the audit, so the trace
and provider usage were irrecoverable. The task is permanently excluded across all six arms with
zero reruns. The maximum retained development cohort is now 659 of 673, or 97.9 percent, still
above the frozen requirement.

After four valid arms, the `luna-medium` command for
`gardener__machine-controller-manager-995` completed successfully but its official trace contained
no graded reward. The worker had terminated and the missing cell was irrecoverable. The task is
permanently excluded across all six arms with zero reruns. The maximum retained development cohort
is now 658 of 673, or 97.8 percent, still above the frozen requirement.

The `sol-max` command for `kubermatic__kubermatic-14462` failed to produce an official graded
reward after four valid arms, and the first `luna-low` command for `giampaolo__psutil-2379` failed
the same way. Both commands completed successfully and both workers had terminated, so the missing
cells and provider usage were irrecoverable. Both tasks are permanently excluded across all six
arms with zero reruns. The maximum retained development cohort is now 656 of 673, or 97.5 percent,
still above the frozen requirement.

A second E2B HTTP/2 control-plane wave then affected ten workers for
`joernio__joern-5591`, `shazow__whatsabi-174`, `pymodbus-dev__pymodbus-2593`,
`moment__luxon-1685`, `icssc__antalmanac-912`, `tailwindlabs__tailwindcss-jit-69`,
`open-telemetry__opentelemetry-go-contrib-3041`, `swc-project__swc-8703`,
`solid__community-server-347`, and `pyca__cryptography-12342`. Every affected attempt terminated
before recording a Docker image, scientific command, or provider call. Together with the five
earlier transport failures, these fifteen tasks remain eligible for exactly one fresh-sandbox
infrastructure retry after the initial controller finishes.

A pre-fit code audit found that the implementation required positive matched task-blind advantage
only after averaging the five split seeds. The frozen rule requires a positive advantage in every
seed. The fitter now applies all four development gates independently to each seed and fails on an
incomplete route vector. This correction happened before development collection or fitting and
before any confirmation outcome was accessed. The remote fit manifest also records successful
sandbox termination only after termination completes, so it cannot claim destruction of fitted
state early. Neither correction changes the frozen candidate grid or selection order.

The full frozen fitter received an E2B performance preflight using synthetic rewards only. It
evaluated all 480 candidates across five seeds for 661 tasks, 489 repositories, and 768-dimensional
features in 424.6 seconds. The 1,586,400 route decisions measured 0.230 ms p50 and 0.457 ms p95,
below the 5 ms gate. The preflight made zero provider calls, accessed no target outcomes, persisted
no fitted state, and terminated its sandbox successfully. Its fitter SHA-256 was
`0485fa5b34d1af9ef751d5763bda9c881ba48a82707135292e21002e6675ea77`, matching the frozen
development fitter. This establishes that fitting and route generation fit within the one-hour E2B
sandbox limit without moving heavy computation onto the Mac.

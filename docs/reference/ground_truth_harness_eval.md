# Ground-truth harness evaluation and optimization

This reference defines how WMH evaluates and improves a harness against real task environments.
Harbor owns task acquisition, environment execution, verification, and trial artifacts. WMH owns
the immutable harness candidate, the host-side model provider, the isolated pi runtime, result
validation, and candidate comparison. No world model participates in this protocol.

The protocol is benchmark-neutral. The first research profile targets the Meta-Harness
Terminal-Bench 2 result, but no reusable WMH type, command, or artifact name depends on that paper.

## Scientific target

The paper reports these Terminal-Bench 2 pass rates with Claude Opus 4.6:

| Agent | Pass rate |
|---|---:|
| Claude Code | 58.0% |
| Terminus 2 | 62.9% |
| Terminus-KIRA | 74.7% |
| Capy | 75.3% |
| Meta-Harness | 76.4% |
| ForgeCode | 81.8% |

The headline improvement is 18.4 percentage points over Claude Code. That is a 31.7% relative lift
over the 58.0% Claude Code score, so shorthand such as "+20%" is ambiguous and must not be used in a
report. The difference from the official Terminus-KIRA leaderboard comparator is only 1.7 points.
The paper's search trajectory instead reports a 64.4% KIRA baseline, making the discovered 76.4%
harness 12.0 points higher in that search context. These are three different contrasts. A result
from a WMH pi seed must be reported against its own matched pi baseline and must not be described as
an 18.4 point reproduction unless the Claude Code control is also reproduced under the same task,
model, and trial locks.

The paper imports the non-Meta-Harness rows from the official leaderboard, so its 18.4 point
headline is an unpaired leaderboard contrast, not a randomized matched control. Meta-Harness ranked
second among the Opus 4.6 agents in that table, behind ForgeCode at 81.8%.

The released artifact invokes `anthropic/claude-opus-4-6` directly. This study is constrained to
Azure and Bedrock. Bedrock exposes the same named model, but changing the serving provider is still
an experimental difference. The primary result is therefore a model-matched, provider-shifted
harness-improvement study, not a bit-for-bit reproduction of the published serving path.

The paper's Terminal-Bench 2 discovery experiment searched and evaluated on the same 89 tasks. Its
released final evaluation uses five trials per task, but that method does not test held-out
generalization. The primary WMH lane instead freezes score-independent discovery and confirmation
partitions before search. It scores candidates only on discovery tasks, selects one winner, and
evaluates that winner once on the confirmation roster sealed from experiment-time search artifacts
and rewards.

The primary estimand is the equal-task expected paired reward delta on repeated executions of those
exact confirmation tasks, conditional on the frozen winner, pi baseline, model routes, attempt
horizon, and execution contract. It is not a claim about future tasks, a task superpopulation, the
discovery complement, or the finite mean over all 89 tasks. The complete planned paired-attempt mean
for each task is one bounded primary observation. Repeated attempts may be arbitrarily dependent
within a task; complete task outcome vectors must be mutually independent. A separately reported
weighted semantic-cluster sensitivity allows arbitrary within-cluster dependence and is
conservative for the same equal-task fixed-roster estimand. Sensitivity failure is inconclusive and
does not replace the primary result.

After selection, the primary winner may also be compared descriptively with pi on all 89 tasks at a
separately frozen attempt count. That full-suite score is useful for locating the candidate relative
to 76.4%, but it is not a reproduction of the paper's adaptive all-89 method because confirmation
tasks did not participate in selecting that candidate.

A separate all-89 adaptive lane would require its own search over all 89 tasks, separately selected
candidate, and locked paper-matched all-89 final. It is deferred from the $15,000 protocol below. A
future lane qualifies as a paper-method reproduction only after all of these parity gates pass:

- the released initial population is behaviorally represented, including the Terminus 2 and
  KIRA-derived scaffolds, and the locked final contains its declared reference seed as a control;
- the proposer matches the paper's coding-agent, model, reasoning, skill, tool, and iteration
  configuration, or a behavioral parity suite establishes an explicitly documented substitute;
- every prior candidate's complete captured source, scores, and raw execution traces remain
  selectively readable through the proposer filesystem, without replacing them with summaries or
  a bounded recent-history window;
- the proposer can perform full program rewrites within the frozen agent interface, rather than
  choosing only from a predefined prompt or parameter mutation template;
- the all-89 two-attempt search matrix, ten-iteration schedule, candidate count, stopping rule, and
  finalist-selection procedure match the released run or independently frozen provenance.

If any gate is unavailable, the lane may proceed only as a method-inspired all-89 adaptive study
and must not be described as reproducing the paper's method. Its scores, proposer workspace,
candidate identity, and budget ledger must remain distinct from the confirmation-valid primary
lane. Confidence intervals from that lane describe repeated-trial variability after selection on
the same tasks; they do not establish generalization beyond the evaluated benchmark. The Bedrock
serving path
would remain a provider-shifted replication even after these gates pass. Freeze the primary lane's
exact task IDs, semantic sensitivity groups, random seeds, attempts, e-value bets, observed lift
floor, and power-gated minimum detectable effect before the first candidate-scoring call.

Primary sources are the [paper](https://arxiv.org/pdf/2603.28052), the
[project page](https://yoonholee.com/meta-harness/), the
[reference implementation](https://github.com/stanford-iris-lab/meta-harness), and the
[Terminal-Bench 2 artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact).
AWS documents the exact worker model in the
[Claude Opus 4.6 Bedrock model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-6.html).

## Question and expected result

The primary question is:

> At a fixed model route, task lock, environment backend, attempt matrix, and per-trial compute
> envelope, how many
> Terminal-Bench 2 percentage points can an optimized WMH harness add over the stock WMH pi
> harness?

The primary endpoint fixes maximum agent turns, output-token budget, wall-clock deadline, command
budget, temperature and other sampling controls, reasoning controls, trial concurrency, agent
concurrency, and provider retry policy across arms. The run configuration digest binds both
concurrency controls because throttling and contention can change timeouts and scores. A candidate
that changes one of those resources is ineligible for the primary winner. It may be reported on a
separate score-cost Pareto endpoint, but its gain must not be attributed to harness quality at
matched compute.

The most likely early improvement is a compact environment bootstrap before the first agent turn.
The paper's winning harness primarily added an initial snapshot of the working directory, `/app`,
installed languages, tools, package managers, and memory. The paper reports two to four saved
exploration turns, while the released artifact README says two to five. This is a hypothesis to
test, not a feature to hard-code.

Directional hypotheses, in decreasing order of confidence:

- The optimized candidate beats stock WMH pi on the matched Bedrock lane.
- On discovery data, the environment-bootstrap ablation accounts for a material share of the
  improvement. This is a descriptive mechanism hypothesis unless a third locked confirmation arm
  is separately funded before search.
- The candidate retains a positive paired delta over pi on the common-task Azure lane, but the
  absolute score differs because Azure and Bedrock do not provide the same model family.
- Matching 76.4% from a stock pi seed is possible but not the planning assumption. The winning
  harness builds on Terminus-KIRA. The paper's documented search run evaluates its KIRA baseline at
  64.4%, while 74.7% is the separate official-leaderboard KIRA comparator used in the final table.
  A result near the headline likely requires adapting that seed or independently recovering
  comparable capabilities first.

A null or negative matched result remains plausible. None of these directional hypotheses is a
stop rule, candidate-selection rule, or substitute for the locked confirmation interval.

Before seeing final results, use these interpretation bands:

| Matched improvement over stock pi | Interpretation |
|---|---|
| less than 1 point | no actionable improvement |
| at least 1 and less than 3 points | promising; the fixed-roster lower bound must exclude zero for a confirmed claim |
| at least 3 and at most 5 points | practically meaningful harness improvement |
| more than 5 points | strong result, requiring leakage and infrastructure audits |

A paper-level claim additionally requires the paper-comparable controls, not just a large delta over
WMH pi.

## Experiment units

One immutable run identity contains:

- the harness execution hash, including file paths and all runtime surfaces;
- the agent implementation name and version;
- the provider, exact model or deployment identity, and non-secret provider configuration;
- the digest-pinned pi runner image;
- the Harbor task-environment backend;
- the Harbor task source, immutable task ref, task checksum, and protected trial-lock digest;
- the exact task and attempt matrix.

Changing any item creates a new run. Results from different identities may be compared, but they
must never be merged into one sample.

A digest-qualified OCI reference can identify a multi-platform index rather than the exact child
manifest Docker executes. The current run identity records the supplied reference but not the
resolved platform and child-manifest digest, so the same identity could otherwise execute different
runner bytes on ARM and x86 hosts. Before scored work, resolve and bind both values after qualifying
that platform; do not guess one host architecture as a generic default.

The experimental candidates are represented using normal `HarnessDoc` surfaces:

1. **pi baseline:** the stock WMH pi harness, frozen before optimization.
2. **bootstrap ablation:** the baseline plus a first-turn environment inventory.
3. **reference-strength control:** required for a paper-result lane, but not yet faithfully
   representable in current WMH. The released KIRA-derived scaffold must first be adapted through
   generic runtime and tool contracts with demonstrated behavioral parity.
4. **searched candidates:** mutations proposed through the existing harness-delta abstraction.

Paper-specific labels belong in experiment configuration and result metadata. They do not belong in
the evaluator, runner, provider, or candidate contracts.

### Reference-scaffold parity gap

The released [Terminal-Bench 2 artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)
uses capabilities that the current WMH-to-Harbor path cannot reproduce by changing only a harness
document. Its KIRA-derived agent has a persistent tmux terminal, batched command submission with
marker-based polling and cleanup, a multimodal image tool, two-step completion confirmation, an
approximately 30 KB observation cap, Anthropic prompt caching, provider retry behavior, and context
summarization. The current WMH Harbor agent exposes bash, file reads, file writes, and submit; each
bash action uses Harbor's buffered execution call; tool observations are capped at 16,000
characters; and the stock pi runtime defaults to 20 turns and 4,096 output tokens without an
equivalent image channel.

Caching, retry, summarization, and completion policy can become candidate-controlled surfaces where
appropriate. Persistent terminal execution, bounded streaming output, multimodal observations, and
process control require benchmark-neutral executor and tool contracts. Until those contracts exist
and parity tests against the released scaffold pass, this implementation can test uplift over stock
pi and the bootstrap ablation, but it cannot claim a faithful KIRA control or a reproduction of the
76.4% paper result. The artifact's behavior and required revisions are documented in the released
[agent](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact/blob/main/agent.py) and
[setup guide](https://github.com/stanford-iris-lab/meta-harness/blob/main/reference_examples/terminal_bench_2/SETUP.md).

## Provider and environment matrix

Use two provider lanes:

- **Bedrock:** Claude Opus 4.6, the paper's fixed worker model. The recommended US profile is
  `us.anthropic.claude-opus-4-6-v1`; freeze that profile and its source region. This is the primary
  model-matched, provider-shifted search lane. Confirm account access and quota before freezing the
  run, and never substitute a newer model mid-run.
- **Azure OpenAI:** one frozen GPT-5.5 deployment, if the subscription has quota, used as a transfer
  lane. It measures whether the harness improvement generalizes across model families. It is not an
  exact reproduction of a Claude result. If GPT-5.5 quota is unavailable, predeclare GPT-5.4 as a
  separate lane rather than substituting it during a run.

Do not pool the lanes. Report a paired delta over the matching pi baseline inside each lane.

Harbor supports two task-environment backends in this protocol:

- **local**, the default, for prebuilt-image Docker development, deterministic debugging, and
  final runs when capacity permits;
- **e2b**, an explicit acceleration option for high-concurrency task execution.

These choices move only Harbor's task environment. The isolated pi runner remains in local Docker
by default, but its backend is configured independently with an exact runner specification.
Docker is required whenever either selected backend is local; a fully E2B task and runner route
does not require a Docker daemon on the evaluator host.

Before any scored E2B run, replay one frozen scripted or golden action sequence against identical
task locks on local and E2B. Compare normalized command outcomes, verifier rewards, final task
state, timeout classification, and environment fingerprints. This deterministic replay, not exact
reward equality from a stochastic model run, is the backend parity gate. If live pi samples are
also used to claim backend equivalence, freeze a separate paired design, equivalence margin,
sample size, and task-clustered analysis with adequate power. Otherwise report those live samples
as descriptive route evidence only. Provider calls remain on the trusted host in both modes.
Provider credentials must not enter task containers, E2B sandboxes, candidate prompts, traces, or
canonical result JSON. Local execution requires a literal prebuilt image and rejects all
task-authored Compose and dotenv sources, host interpolation, MCP/skills sources, and run-level
imports, mounts, overlays, host variables, backend kwargs, or extra hosts before Harbor creates a
job. For both backends, a local dataset is copied into a content-addressed, read-only WMH snapshot,
revalidated there, and only that snapshot path is given to Harbor. The broader task audit still
reports only variable names and source locations; it never resolves or logs credential values.
The source must be a private checkout owned by the evaluator user on one filesystem: group/world
writable entries, hardlinks, symlinks, special files, and cross-filesystem descendants fail closed.

### Python scoring boundary

`HarborHarnessScorer` is the synchronous bridge from one explicit Harbor task selection to the
generic harness-search score contract. The dataset selection and `task_ids` must be the same
ordered, literal list. Globs, exclusions, random task limits, request-time subsets, and
request-time attempt changes are rejected. Scored search accepts only local dataset paths that WMH
can inspect before Harbor job creation. Harbor 0.18 dereferences symlinks while copying remote/Git
tasks, before WMH can validate the downloaded tree, so registry, package, and remote acquisition
are rejected by the ground-truth evaluator under both Docker and E2B until a symlink-preserving
acquisition boundary lands. Dataset-qualified task keys from a frozen qualification
manifest bind each selection to its Harbor source, identity, and task checksum across candidates.
Qualification also freezes the executed environment definition. Local runs attest the Docker
daemon platform plus every Compose service's immutable image ID and image platform. Ephemeral
container, trial, and project identities are deliberately excluded. Every scored attempt must
reproduce the per-task qualification digest.

WMH does not use Harbor 0.18's alias-based E2B create path for scored runs. Its trusted adapter
serializes a content-and-resource-keyed build registry, records `BuildInfo.template_id` and
`BuildInfo.build_id` only after a completed build, and creates each sandbox with the exact
`<template_id>:<build_id>` reference. It verifies the returned template, resources, platform,
network and lifecycle policy, excludes ephemeral sandbox identity from the stable environment
digest, and writes an owner-bound resource receipt after metadata reconciliation proves the
sandbox absent. Valid and candidate-damaged E2B cells are rejected unless that receipt is terminal
and bound to the attested launch configuration.

A task `storage_mb` value is a launch-time minimum, not an E2B template-build input. The E2B SDK
does not accept disk size when building the exact template, so different task minima reuse the same
content, CPU, and memory keyed build. The requested minimum is instead bound into the launch digest.
After create, WMH reads E2B's provider metrics, interprets `disk_total` as bytes, retries only an
initially empty metric series across a fixed ten-second polling window with bounded requests, and
requires the conservative observed total to cover the requested MiB before accepting the sandbox.
The stable evidence labels this quantity as `provider_reported_total`; it is root-disk allocation,
not a claim about currently free bytes. Missing or malformed metrics fail
closed and use the ordinary owner-bound sandbox cleanup path.

The exact E2B adapter preserves Harbor's full-day sandbox lease because one environment spans
setup, agent execution, and shared verification. Its timed-resource account reserves that complete
provider TTL before create, then settles only the observed lifetime after cleanup. This prevents an
individually valid agent or verifier timeout from expiring the shared environment between phases.

Local Docker has no portable per-container disk quota in this adapter. For a task storage request,
WMH runs the same sanitized writable-inode and POSIX `df -Pk` probe used by command health checks and
requires that the task's current filesystem report at least the requested available KiB. The stable
attestation explicitly identifies this as shared-task-filesystem admission and records that it is
not provider enforced; it omits the volatile free-block count. A pass therefore proves conservative
headroom at admission time only. Parallel cells can consume the same backing store afterward.

```python
from harbor.models.job.config import DatasetConfig

from wmh.evals.harbor.config import HarborJobSpec
from wmh.evals.harbor.scorer import HarborHarnessScorer
from wmh.harness.cost import SearchComponentRole
from wmh.harness.create import search_harness
from wmh.providers.receipt import ProviderResponseIdentity

task_ids = ("task-a", "task-b")
# Use the exact QualifiedHarborTask objects published by roster qualification. They are the single
# source of task IDs, content keys, environment digests, E2B build IDs, and resource classes.
qualified_by_id = {task.task_id: task for task in qualification_roster.tasks}
qualified_tasks = tuple(qualified_by_id[task_id] for task_id in task_ids)
worker_response_identity = ProviderResponseIdentity(
    provider=worker_provider.kind,
    response_model=expected_response_model,
    system_fingerprint=expected_system_fingerprint,
)
job = HarborJobSpec(
    job_name="discovery",
    jobs_dir=".wmh/evals/harbor",
    datasets=[DatasetConfig(path="/frozen/tasks", task_names=list(task_ids))],
    n_attempts=2,
)
with HarborHarnessScorer(
    job_spec=job,
    provider_config=worker_provider,
    response_identity=worker_response_identity,
    reference_harness=baseline,
    qualified_tasks=qualified_tasks,
    reward_key="reward",
    cost_runtime=search_cost_runtime.for_component(SearchComponentRole.SCORER),
) as scorer:
    result = search_harness(
        "discovery",
        baseline,
        scorer,
        proposer,
        iterations=5,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        cost_binding=search_cost_runtime.binding,
    )
```

When this scorer is passed to `search_harness`, set `screen_proposals=False` and
`confirm_narrow_vetoes=False`. Every candidate then receives exactly the frozen matrix. The
reference harness fixes runtime kind, turns, output tokens, temperature, effective tools, and the
per-turn deadline. A candidate that changes that compute envelope is ineligible.

The score plan also freezes provider-reported route evidence. Azure and OpenAI routes must name the
exact served response model before search, plus the system fingerprint when that endpoint exposes
one. Bedrock Converse exposes neither field, so its committed response identity contains explicit
null values. Receipt drift is rejected before a scored arm is admitted.

`environment_backend="local"` selects local task and pi-runner infrastructure only. Azure or
Bedrock inference remains paid and still requires the same complete `SearchCostBinding` and shared
durable budget authority as E2B execution.

When any search component creates E2B resources, the top-level cost binding also commits one
path-free external dispatch-rate binding. Pass the corresponding authority to the proposer
`AgentProject` and every E2B scorer. Project sandboxes, task environments, and pi runners then
acquire from the same durable four-per-second gate before each provider create.

The local policy prevents Harbor from building task Dockerfiles or consuming task-authored Compose
host capabilities, but it is not proof that an arbitrary container image or verifier is safe.
Every scored dataset must therefore be frozen by content, reviewed as executable code, and limited
to audited task sources and resolved image digests before credentials are loaded or a paid run is
approved.

Harbor job metrics and remote dataset metadata can also declare `uv-script` metrics, while package
datasets can provide a dataset-level `metric.py`. Harbor runs those scripts as host subprocesses
with the evaluator environment. WMH rejects all three executable-metric entry points from metadata
before Harbor job creation and accepts only Harbor's non-executable built-in aggregate metrics.
Canonical per-trial rewards remain the input to trusted statistical analysis.

### Harbor 0.18 command-execution limit

Harbor 0.18's cross-backend `BaseEnvironment.exec` returns a fully buffered result. It exposes no
portable output stream, host-side byte cutoff, process handle, or confirmed kill operation. Every
WMH task-tool call now receives the absolute pi-turn deadline. The Harbor bridge bounds its own
wait and Harbor's integer execution timeout to the remaining turn budget, and places an earlier
deadline and output cap inside the task. A task-side exit 124 remains a candidate tool observation;
exhaustion of the caller-owned turn deadline remains a gradeable candidate timeout. Raw exit 137,
SIGKILL/OOM, ENOSPC, or filesystem-postcondition loss is not enough to assign ownership: parallel
local Docker cells share host memory and backing storage, and GNU `timeout --kill-after` can itself
produce exit 137. Those signals therefore carry `environment_confirmation_required` and cannot
enter optimizer selection. In addition to bounded head-and-tail command evidence, the bridge
establishes a healthy free-block and writable-inode baseline, then runs a fixed POSIX `df -Pk` and
inode-creation postcondition after each command. Falling below the frozen 128 MiB reserve or losing
the writable-inode postcondition requests the same fresh-cell confirmation. The task image must
provide the same Bash and POSIX utilities already required by the tool executor; qualification
exercises this probe before scoring.

These controls handle ordinary commands and frozen task images. They observe only the command's
default filesystem. Exhaustion of another verifier mount remains ambiguous. They are also not a
trusted kill boundary against a hostile root candidate: root can replace the task-side shell or
utilities, detach descendants, or otherwise defeat a mutable in-container supervisor, while
Harbor exposes no portable descendant-kill or host-side filesystem proof.

The next execution-boundary slice must implement exactly one explicit confirmation attempt for
the typed `environment_confirmation_required` outcome. It must start a fresh sandbox, reproduce
the same candidate and benchmark cell, receive its own manifest entry and attempt identity, and be
charged to the external-resource and spend ledgers. It must not use Harbor's hidden retry path.
Only a second matching environment loss under the frozen confirmation policy may be promoted to
candidate damage; a nonmatching result invalidates the pair as infrastructure evidence.

Before paid search, do one of the following:

1. add a Harbor streaming-exec and kill-handle contract, then enforce the byte limit in the trusted
   evaluator; or
2. run each Harbor evaluator worker inside a disposable, externally memory- and PID-limited worker
   so a poisoned buffered result can only invalidate that worker.

Also freeze and verify task-image digests. Harbor's task checksum covers the task directory and its
configuration, but a mutable Docker image tag names content outside that directory. Use non-root
task users where the benchmark permits, but do not silently change the official task user or image
when producing a paper-comparable result. Until the stronger boundary exists, describe task-side
caps as best-effort qualification controls, not proof of host-memory safety.

### Harbor 0.18 artifact-recovery limit

WMH serializes each job name with an interprocess execution lease and atomically replaces Harbor's
frequently updated root `result.json`. The temporary result is a job-qualified sibling under the
jobs directory, not a file inside the job root, so a process killed between write and rename cannot
make strict job-layout validation reject otherwise valid evidence. Such an orphan is preserved and
ignored during resume. The CLI also refuses to use the active job's lease path as canonical output,
which prevents output publication from replacing the locked inode. These controls prevent two
resume processes from paying for the same missing cells and prevent a process interruption during a
root progress update from truncating the last valid root result.

A parseable cancelled Harbor result is terminal evidence, not a missing cell. WMH reports it as
`cancelled`, preserves its artifacts, and does not claim the same job can rerun it. Until an atomic
per-attempt ledger exists, an operator who explicitly wants another attempt must use a new job name.

Harbor 0.18 still writes the root job config and lock plus each trial config, lock, and result
directly. An interruption during one of those one-time writes can leave an unreadable artifact even
when some paid work completed. The evaluator deliberately rejects that job instead of guessing at a
job ID, task lock, or score, and it performs no automatic reconstruction. Before paid search, make
those Harbor writes atomic upstream or prove a write-ahead snapshot and restore procedure with
kill-injection tests. The spend ledger must count any orphaned provider call against the phase cap.

Normal backend teardown is not crash recovery. Harbor's delete-on-stop setting cleans up task
environments when the owning trial reaches its teardown path, but a process or host interruption can
release the WMH job lease while leaving local Compose projects or remote sandboxes alive. Before paid
concurrent work, add an evaluator-owned external-resource ledger that records the backend, opaque
resource ID, benchmark cell, creation time, and teardown state without credentials. Startup must
reconcile every pending resource ID with its backend, remove confirmed orphans, and refuse new paid
work when absence cannot be proved. Exercise interruption after environment creation and before
result publication in local and remote kill-injection tests. Attribute both observed orphan charges
and the conservative cost of any unproved resource to the active phase envelope and hard spend
ledger.

### Provider failure and deadline limits

The current worker-provider interface is synchronous and does not accept a per-request deadline.
The pi turn deadline therefore cannot preempt an Azure or Bedrock call already in flight, and
cancelling the surrounding worker thread does not cancel the billable provider request. Before
paid search, add a deadline-aware provider contract that passes the remaining turn budget into the
SDK request, or run the provider call in a cancellable disposable worker with an externally proved
termination boundary. Do not add an untracked background executor that can leave paid calls alive.

Azure/OpenAI request failures become candidate-owned invalid-request outcomes only for frozen
candidate codes or structured candidate parameter roots, after credential, deployment/model
route, throttle, timeout, transport, and server precedence. A known 422 `invalid_request_error`
also covers tool-call ordering failures without a parameter path. Bedrock `ValidationException`
and local botocore parameter validation cross the boundary only through anchored provider templates
or SDK-authored candidate parameter roots. Unknown request shapes fail closed as infrastructure.
Those typed candidate failures are valid zeroes. Auth, route, throttle, timeout, transport, 5xx,
and unknown errors produce retry-required run health. Raw provider text is never copied into
candidate-visible or canonical evidence. `HarborJobSpec` still rejects every nonzero retry
configuration because Harbor 0.18
overwrites the final trial result without retaining complete attempt usage and exceptions. Before
enabling paid retry, add an atomic attempt ledger and validate the taxonomy with live probes on the
frozen Azure and Bedrock routes.

Current agent evidence records input and output tokens for completed calls, but not cache-token
detail or authoritative provider cost. Each canonical usage field carries an `exact`,
`lower_bound`, or `unavailable` status. When a provider call fails without authoritative failed-call
metering, already observed totals remain as lower bounds rather than being mislabeled as exact zero
or exact partial totals. The canonical result keeps unavailable cache tokens and provider cost
missing rather than estimating them from a static price table. The spend ledger must reconcile
provider usage and contracted billing outside the task environment before it admits the next paid
block.

### Time-budget parity

The released reference setup uses `HARBOR_TIMEOUT_SECONDS=28800`, an eight-hour outer limit for an
agent trial. The `wmh harness eval` CLI currently defaults `--turn-timeout` to 300 seconds for its
single pi run. That five-minute default is suitable for development canaries, but it cannot be used
for a paper-strength lane. Before qualification, explicitly freeze the Harbor trial and agent
timeouts, WMH pi-run timeout, remaining-budget command timeout, provider-request deadline, maximum
agent turns, and scheduler deadline. Record them in run evidence and validate their ordering so an inner limit
cannot silently truncate an otherwise eight-hour-compatible trial. Eight hours is a comparability
ceiling, not permission to leave billable provider requests unbounded.

Before qualification, complete these provider checks:

- submit the required Anthropic model-access form and verify the exact Bedrock profile once outside
  a scored job;
- read the account's actual Bedrock Service Quotas in the frozen source region rather than assuming
  the published defaults;
- record the Azure resource, region, deployment name, underlying model version, API version,
  deployment type, content-filter policy, and assigned quota;
- attest the Azure deployment configuration immediately before and after each scored phase, and
  stop if its underlying model revision changes; the current WMH trace meters tokens but does not
  yet retain the response-reported served model revision;
- set concurrency below the measured quota in both lanes so throttling cannot become
  candidate-dependent;
- snapshot the operator's live contracted prices. Public list prices and the paper's historical
  cost are planning inputs, not billing truth.

Relevant provider references are the AWS
[model-access guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html),
[Bedrock quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html), and
[geographic inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/geographic-cross-region-inference.html),
plus Microsoft's [Azure model catalog](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure),
[quota guide](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota), and
[default safety policies](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/default-safety-policies).

## Phased protocol

### Phase 0: freeze inputs and controls

1. Record the exact WMH commit and clean-tree status, Harbor version, and Terminal-Bench 2 ref,
   then resolve all task checksums. The released setup identifies Terminal-Bench 2 commit
   `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` as the paper submission target and terminal-bench
   package commit `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b` as the evaluator dependency. Audit and use
   those snapshots for every all-89 comparison unless provenance evidence establishes different
   refs. If evaluator workers are containerized, freeze their image digest too.
2. Freeze the pi baseline and, if used, the reference-strength control.
3. Freeze one Bedrock Claude Opus 4.6 inference profile ID and one Azure deployment plus API
   version.
4. Complete the power and partition gate below, then freeze the discovery and confirmation split
   without inspecting benchmark rewards or candidate scores.
5. Select one reward key, expected to be `reward` for Terminal-Bench 2, in experiment analysis only.
6. Register the $15,000 hard ceiling, initial phase caps, and the finite cost-only reallocation rule
   that will freeze immutable final caps after qualification and before search.

#### Sealed confirmation boundary

The split manifest is a control-plane artifact, not proposer context. Before materializing any
search workspace, create a discovery-only view and prove that the proposer cannot read or fetch:

- confirmation task IDs or names, instructions, checksums, source trees, Dockerfiles, tests,
  verifiers, solutions, images, locks, manifests, Harbor jobs, results, or traces;
- the unsplit benchmark checkout, Harbor registry cache, package cache, or object store from which
  confirmation content can be reconstructed;
- task-level historical outcomes or difficulty labels used to infer confirmation membership.

Run the proposer in a dedicated workspace containing only candidate source, its frozen
domain-general instructions, and discovery artifacts. Deny network access or allowlist only
services that cannot retrieve the public benchmark or its mirrors. Restricting writes is not a read
boundary. Before and after every proposer turn, validate a content allowlist and a workspace digest;
any unexpected file, path, task marker, or network capability invalidates the search before more
candidate calls are admitted. Keep the split seed, confirmation IDs, and score-independent group
metadata in a
separate operator-owned location that is never mounted into that workspace.

Family metadata used by the split generator must have frozen provenance and must be independent of
WMH, candidate, and task-level leaderboard rewards. Only the sealed control plane assigns tasks to
strata and partitions. Aggregate discovery stratum counts may be published to the proposer when
needed, but confirmation membership may not. Public availability and possible model pretraining on
Terminal-Bench remain limitations. This seal establishes only that search receives no confirmation
artifacts or rewards through experiment-time storage, tools, or network access. It cannot show that
confirmation content was absent from model training, prevent a proposer with a memorized public task
roster from inferring the complement of the discovery set, or rule out consequential pretraining
contamination. Such leakage invalidates the held-out selection claim even though the statistical
estimand is conditional on the exact confirmation roster. The seal cannot verify model-training
provenance, so the report must state this limitation explicitly.

The final evaluator may of course expose one confirmation task instruction and environment to the
frozen worker candidate while that task is running. It must not expose confirmation content to the
search proposer or return final evidence until candidate identity is frozen. The post-run string
and source audit is defense in depth, not a replacement for this access boundary.

#### Power and partition gate

Before any benchmark reward or candidate score is observed, write and freeze a power-design
manifest. It must predeclare the exact inferential target, confirmation roster size, per-lane attempt
counts, one-sided alpha, fixed e-value mixture, observed equal-task lift floor, candidate-selection
rule, weak-null and target-alternative data-generating models, nuisance ranges, target minimum
detectable effect, requested power, Monte Carlo alpha and replication count, semantic sensitivity
groups, analysis code identity, seeds, and minimum discovery constraints. Baseline pass rates,
within-task attempt dependence, paired-arm dependence, cross-lane dependence, and failure processes
must come from external evidence or a frozen plausible range, not WMH candidate outcomes.

The primary estimand for lane `m` is
`N^-1 sum_t E[X_t,m | exact roster and frozen execution contract]`, where `X_t,m` is the complete
planned paired-attempt mean delta for exact confirmation task `t`. The one-sided primary null is that
this equal-task conditional mean is at most zero. The complete task outcome vectors must be mutually
independent, while attempts and arms inside one task vector may be arbitrarily dependent. No
exchangeability or future-task sampling assumption is used or claimed. Candidate selection on the
discovery complement means this result also does not identify the selected candidate's finite mean
over all 89 tasks.

Use a locked simulator that generates the complete task vectors and runs the exact production
analysis, including the observed effect floor and all-lane intersection-union decision. Its target
alternative uses one deterministic semantic-group effect assignment for the fixed roster. That
assignment is shared across lanes and every replicate; each lane's additive scale is solved against
the realized fixed assignment so its clipped equal-task effect reaches the declared target within a
forward machine-precision bound. A positive target may never collapse to zero. Task and lane
outcomes are sampled independently conditional on those frozen probabilities. Randomly
redrawing semantic-group effects per replicate would change the fixed-roster estimand and is not
permitted.

It must produce the predeclared number of memberwise `weak-null` configurations and
`target-alternative` replicates with no optional stopping. Let `R_m` be lane `m`'s exact primary
rejection event and `R = intersection_m R_m` the all-lane decision. For every designated null lane
`j`, `R` is a subset of `R_j`, so `P(R) <= P(R_j)`. The frozen DGP also makes lane `j`'s marginal
invariant to every other lane's nuisance effect through independent conditional lane sampling and
fixed lane-specific probabilities. The memberwise null simulator therefore draws only lane `j` at
its zero-effect boundary and records `R_j`, conceptually setting every other Boolean member decision
to pass. This is a conservative upper bound on the composite-null IUT rejection probability, not a
claim that any finite other-lane effect makes a stochastic decision pass surely. If other-lane
nuisance can alter the designated lane's marginal, this bound and the power gate are invalid. The
manifest freezes this contract as `memberwise-marginal-conservative-upper-bound-v1` with
`all-other-member-decisions-pass`. The executable power gate
rejects missing, duplicate, extra, or simulation-digest-drifted replicates. It uses preregistered
one-sided exact Clopper-Pearson bounds certified outward by directed-rounding, high-precision
binomial tail sums; SciPy quantiles are only initial guesses. It allocates the Monte Carlo error
across all memberwise nulls with directed-downward floating-point division, records that exact
allocation, takes their worst simultaneous upper bound, and requires that bound not to exceed the
maximum type-I error. The target-alternative rejection-rate lower bound must reach the requested
power. Its durable report embeds the full public-safe simulation manifest in the frozen gate design
and cross-validates its digest, complete lane-null roster, numerical runtime, target effect,
evaluation-design digest, and replication horizon. It binds the complete canonical
trial evidence by digest, and binds every memberwise count and bound plus the aggregate counts,
rates, bounds, and decisions with a report digest. Reloading must recompute and validate every
derived value under the exact frozen runtime, executable, source, and schema identities. The report
is durable but deliberately environment-bound; transfer to a different environment fails closed.
Passing supports only that frozen MDE and data-generating assumptions.

Three percentage points and 80% power remain desired planning values, not established operating
characteristics. Do not call the current design powered for 3 points until the locked simulation
manifest exists and its executable gate passes. Before scoring, enumerate feasible integer
discovery/confirmation allocations and attempt horizons, preserve the minimum discovery matrix, and
choose exactly one score-independent route: the 3 point design if it passes; a larger predeclared MDE
that passes; or stop and add independently funded compatible tasks. Record the simulator, manifest
digest, inputs, outputs, chosen MDE, task counts, analysis seed, exact task IDs, semantic groups, and
matrix digest. Never lower the requested power or change the route after observing scores.

The study-profile recommendation for the forthcoming manifest is 59 confirmation tasks, 20 planned
attempts per task, lane, and arm, a 3 point observed floor, and a 10 point powered MDE. Freeze the bet
mixture exactly as `f=1/4, weight=1/16`; `f=1/2, weight=1/16`; and `f=1, weight=7/8`. The power gate
must bind both the simulator digest and the exact paired-evaluation design digest containing this
roster, attempt matrix, mixture, and floor. A 50,000-replicate calibration reported individual-lane
power of 94.2% to 94.6%, all-three-lane power of 85.9%, and heterogeneity sensitivities from 80.4% to
91.3% at a 10 point effect. Those figures are study-design inputs, not accepted executable evidence
until the simulator artifact and complete digest-bound trials are committed. They depend materially
on the declared complete-task-vector independence and near-zero residual attempt ICC: the reported
all-lane power was 80.4% at ICC 0.01 and 58.3% at ICC 0.05. The weighted semantic-cluster sensitivity
is expected to be underpowered and remains non-gating.

The reusable simulator contract lives in `wmh.evals.power`. Build its private
`PairedPowerTaskProfileManifest` only after partition genesis has frozen the complete private split
and exact confirmation design. This does not open confirmation identities to search. The profile
contains task identities, strata, semantic groups, and fixed lane nuisance rates, so it stays in the
private experiment control store. Creating
`PairedPowerSimulationManifest` binds only digests of that profile and its task metadata into the
portable simulation identity. It also binds the exact v5 evaluation-design digest, lane attempts,
bet mixture, effect floor, DGP and effect atoms, residual attempt ICC, seed, fixed replication
horizon, deterministic chunk size, simulator schema, and simulator and paired-analysis source
bytes. The identity additionally freezes the exact Python implementation and version, executable
digest, cache tag, operating-system release and build, machine, NumPy, SciPy, Pydantic, and
Pydantic-core versions and installed-distribution `RECORD` digests, and explicit RNG, clipped-effect
solver, and Clopper-Pearson certifier implementation identities. Chunk generation and gate
evaluation fail closed if the current numerical runtime or executable source/schema differs.

Run every prescribed memberwise null and target chunk with `run_paired_power_chunk`, or use
`resume_paired_power_simulation` for local checkpoints. Chunk identities derive from the frozen
scenario, null member where applicable, and inclusive replicate range. Merge rejects missing,
duplicate, extra, or digest-drifted ranges. Resume opens every directory component without following
symlinks and holds the final directory descriptor, requires the current user to own a mode-`0700`
checkpoint directory and mode-`0600` regular artifacts,
rejects symlinks and special files without blocking, detects inode replacement during reads, and
publishes each chunk immutably with an exclusive same-directory link. Existing artifact names are
never overwritten. The merged `PairedPowerTrialArtifact` stores complete decision bitsets and chunk
digests but no task identities, strata, group names, baseline rates, or outcomes. Persist and reload
it with `write_paired_power_trial_artifact` and `load_paired_power_trial_artifact`; the loaded compact
artifact can be passed directly to `evaluate_paired_power_gate` without expanding all Pydantic trial
records. The gate streams the bitsets into the same canonical per-replicate evidence identity used
by expanded trials, so either representation produces the same evidence and report digests.

The checked-in tests use only a four-task synthetic profile and are explicitly not study evidence.
Generate the scored 50,000-replication artifact per scenario only after the private 59-task design
and profile are frozen, and finish its executable power gate before any paid discovery call. Never
commit the private task profile, split opening, chunk directory, or an artifact that exposes sealed
identities. A public report may retain the compact identity-free trial artifact after a string audit
and after its simulation manifest digest has been committed to the experiment chronology.

The same manifest must freeze the Azure transfer task and attempt matrix, or a finite conditional
ladder of matrices selected only by measured per-cell cost and quota. If a ladder is used, record
its thresholds before qualification, keep candidate rewards hidden, and resolve it from cost and
infrastructure evidence only before Bedrock search begins. Azure scores must never choose the
transfer matrix.

If no feasible roster reaches the preregistered power for 3 points, any observed 3 point lift remains
an effect-size description and not a powered-MDE result. A locked all-89 paired estimate may be added
only as a benchmark-descriptive endpoint; it does not change the fixed-roster primary estimand.

### Phase 1: zero-cost and low-cost qualification

Run schema, manifest, resume, candidate-failure, timeout, verifier, and credential-isolation tests.
Then freeze a cross-family qualification matrix containing only discovery tasks or purpose-built
non-benchmark fixtures. Confirmation tasks may not appear in qualification. Run:

- the same deterministic scripted or golden action replay on local and E2B with identical task
  locks;
- the bootstrap ablation on local;
- pi on one Azure and one Bedrock route, using local by default.

One live model attempt per selected task is a route-reachability and failure-classification canary,
not evidence of reward parity. Freeze the task IDs and an absolute allowed-failure count before
running the matrix. Stop if canonical results contain unclassified failures, if deterministic local
and E2B replay differs in a predeclared normalized command outcome, final task state, verifier
reward, timeout classification, or environment fingerprint, or if infrastructure failures exceed
the allowed count after any retries admitted by the atomic attempt ledger. Without that ledger, do
not retry and count the first-attempt failure. Report the live-sample percentage too, but do not
use it as the backend equality rule. A backend-equivalence claim from live pi requires the
separately powered distributional design above.

Use pi, not a mutable candidate, for the Azure route canary. Expose only provider identity, quota,
latency, cost, and typed infrastructure evidence needed to resolve a predeclared transfer-matrix
ladder. Seal every task instruction, reward, trace, tool transcript, verifier outcome, and
task-level usage value from both the proposer and candidate-selection process until the transfer
matrix and primary Bedrock winner are frozen. If the canary system cannot separate those views,
defer the Azure canary until after winner freeze.

### Phase 2: establish matched baselines

Run the stock pi baseline on the frozen discovery matrix with two attempts per task in the primary
Bedrock lane. Run the reference-strength control there only after the runtime and tool parity gate
passes. Freeze these results before search. Do not run the scored Azure baseline until the Bedrock
winner is frozen. Do not run or reveal baseline scores, traces, or verifier outcomes on confirmation
tasks before the primary candidate is selected; those cells belong to the blocked final.

This phase calibrates stock WMH pi on discovery tasks relative to the published Claude Code,
Terminus 2, and KIRA results. Those leaderboard values are not matched controls, and a discovery-only
location is not an all-89 score. The exact descriptive comparison is deferred until after selection.

### Phase 3: search

Use the Bedrock lane for primary search. The frozen discovery matrix must exclude every confirmation
task, and task filters must be checked against the split manifest before each candidate job. Begin
with the pi baseline, bootstrap ablation, and optional reference-strength control. Each iteration
should:

1. expose the proposer to candidate source, canonical scores, bounded traces, and explicit cost;
2. propose normal harness deltas, with no direct mutation of the evaluator or verifier;
3. evaluate the new candidate on the frozen discovery matrix;
4. retain a small Pareto population over score, cost, latency, and complexity;
5. record every rejected and retained candidate by execution hash.

Freeze the proposer implementation, provider route, model revision, prompt, reasoning controls,
tool access, history policy, maximum proposals, and cost accounting before search. Proposer calls
consume the same $15,000 ceiling as worker calls.

Also freeze one deterministic primary-winner rule. First reject candidates that violate the exact
matched-compute envelope, omit planned cells, exceed a predeclared infrastructure or candidate-
failure ceiling, or fail the leakage audit. Among the remaining candidates, choose the highest
discovery mean reward. Break an exact score tie by lower total input-plus-output tokens, then lower
canonical serialized harness bytes, then lexicographically smaller execution hash. If token
metering is incomplete for any tied candidate, skip the token tie-break for every member of that
tie. The Pareto archive remains useful for secondary analysis, but an operator may not manually
choose another frontier point after seeing scores. A different tradeoff requires a separately
declared exploratory winner and cannot replace the primary winner.

Run candidates concurrently through Harbor, subject to provider quotas and the dollar gate. Do not
run multiple statistical attempts for the same task concurrently if provider or environment rate
limits could create candidate-dependent throttling.

The official released search configuration uses all 89 tasks, two trials per task, concurrency 50,
and reports roughly $500 and four to six hours per iteration with Opus 4.6. Treat those figures as a
planning reference only. Recalculate from live Azure, Bedrock, Harbor, and E2B prices before launch.

At the primary search limit, apply the frozen winner rule and freeze its winner without opening
confirmation evidence. The $15,000 protocol does not start an all-89 adaptive search. A future,
separately funded lane must use a fresh workspace and budget ledger, meet every paper-method parity
gate above, and select its winner independently. The primary winner cannot be relabeled as an
all-89 winner, even if it later receives an all-89 descriptive score.

The primary winner must be frozen before any Azure candidate comparison used for the transfer
endpoint. Do not select, revise, or rank it using Azure results. An Azure-guided iteration belongs in
a separately budgeted exploratory lane and cannot support the cross-provider transfer claim.

### Phase 4: locked confirmation

Each lane's winner must already be frozen. Make no further harness changes after observing its
locked results.

- Primary confirmation: pi baseline and the primary winner on the frozen confirmation partition at
  the attempt horizon selected by the locked power-and-cost design.
- Full-suite descriptive comparison: if it is fully funded, extend that same blocked final matrix
  to the discovery tasks so pi and the frozen primary winner receive the same frozen attempt horizon
  on all 89 tasks with the fixed primary model. Compute the primary endpoint from only the prespecified
  confirmation subset and the descriptive score from all 89 tasks. Do not rerun confirmation cells
  in a second full-suite job. This comparison is not an adaptive-method reproduction. Add the
  reference-strength control only if the final-evaluation envelope still covers every paired
  primary cell. If all 89 tasks do not fit, run the confirmation matrix and omit the full-suite
  descriptive score.
- All-89 adaptive comparison: deferred and unfunded in this protocol. It requires a new budget that
  covers the complete frozen search plus a separate paper-matched paired final before its first
  candidate call.
- Transfer lane: the already-frozen Bedrock winner and its matching pi baseline on the frozen Azure
  deployment. Before search, bind the exact task IDs and attempt count to a score-independent
  power-and-cost design and its matrix digest. Do not change the matrix after observing any Azure
  candidate comparison. Use the same confirmation
  tasks as Bedrock, or a predeclared score-independent subset of them, so task mix does not masquerade
  as provider transfer. Treat transfer as secondary unless its manifest also freezes a powered
  criterion. Report the Azure paired delta and the Azure-minus-Bedrock delta on the common tasks;
  do not claim that a fraction transferred without a predeclared retention threshold and interval.

The bootstrap mechanism contrast remains discovery-only and descriptive under this budget. Do not
claim a confirmed fraction of improvement attributable to bootstrap unless a third pi, bootstrap,
winner confirmation arm and its share estimand are fully funded and frozen before search. That
addition must not reduce or unpair any primary pi-versus-winner cell.

The current runnable matrix has no Claude Code arm. The paper's 18.4 point contrast is therefore
external calibration only. A matched claim against Claude Code requires a generic external-agent
control, the same locked task and attempt blocks, and a newly costed budget before launch.

If the budget cannot fund every cell, preserve the pi-versus-winner paired cells first. Drop a
secondary seed or reduce transfer attempts before creating an unpaired primary matrix.

Execute each final comparison as task-and-attempt blocks spanning both arms. Randomize which arm
starts first inside each block, keep per-arm concurrency equal, and place both arms in the same
provider time window. Do not run the complete pi matrix and then the complete winner matrix. This
blocking limits service drift, quota incidents, and policy changes from becoming harness effects.

### Phase 5: audit and report

Before claiming an improvement:

- inspect all task-level discordant pairs;
- search candidate source and generated prompts for task names, solutions, test constants, and
  benchmark-specific branching;
- review a stratified trace sample, including all candidate failures and verifier anomalies;
- report infrastructure, timeout, incomplete, and unclassified counts separately from task score;
- reproduce the winner from its task locks and execution hash in a clean job directory;
- publish negative ablations and total spend, not only the winning score.

## Analysis contract

For a cell that finishes without a candidate-owned failure, its analysis outcome is the selected
binary verifier reward. Candidate-owned invalid provider requests, agent/task timeouts, and typed
task-container destruction or exhaustion have primary outcome zero. Preserve any later verifier
reward and the candidate failure kind as diagnostic evidence. A verifier, provider, environment,
runner, or unclassified failure with retry-required or unknown run health is not a zero and
invalidates the whole paired block. A predeclared, fully ledgered recovery must rerun both arms in
fresh isolation without inspecting either score; it may not fill only the failed arm. The timeout
sensitivity analysis substitutes Harbor's valid post-timeout verifier reward for the primary zero;
it may not replace the primary result after scores are observed.

For lane `m`, first compute each complete task observation
`X_t,m = A_m^-1 sum_a (R_candidate,t,m,a - R_pi,t,m,a)` over the exact planned attempts. Then report
the observed equal-task delta `D_m = N^-1 sum_t X_t,m`. The primary inferential estimand is
`theta_m = N^-1 sum_t E[X_t,m]`, conditional on the exact sealed roster and frozen execution
contract. Every task has equal weight. This is repeated-execution evidence for these tasks, not
future-task or all-89 generalization.

For the one-sided null `theta_m <= u`, each preregistered bet fraction `f` uses
`E_f(u) = product_t [1 + f (X_t,m - u)/(1+u)]`; a frozen convex mixture over bet fractions is the
lane e-value. Complete task vectors may have different distributions. Under the weak equal-task
null, mutual task independence factors the expectation and AM-GM bounds it by one. The reported
primary p-value is `min(1, 1/E(0))`. Inverting `E(u) > 1/alpha` gives a conservatively rounded
one-sided lower bound. Compute the pass decision from the exact rational rejection
`E(0) * alpha > 1`, not by reparsing the downward-rounded float endpoint: a mathematically positive
endpoint can be too small for a positive float representation. Each lane uses the unadjusted
one-sided alpha of 0.05. The all-lanes claim is an intersection-union test: every lane must reject
its zero null, so no cross-lane independence or alpha division is required. Every lane's observed
`D_m` must also meet the separately frozen 3 percentage-point floor.

Report a dependence sensitivity using the predeclared semantic groups. For group `g`, let `X_g,m`
be its task mean, `w_g = n_g/N`, `w_max = max_g w_g`, and `c_g = w_g/w_max`. Its bet factor is
`1 + f c_g (X_g,m-u)/(1+u)`. Under the same equal-task null,
`sum_g c_g(E[X_g,m]-u) <= 0`; independent groups plus AM-GM therefore give a conservative e-value
for the same equal-task estimand while allowing arbitrary dependence inside each group. Report its
p-value and inverted lower bound per lane. A nonpositive sensitivity lower bound is inconclusive,
does not fail the primary result, and must never be presented as evidence that semantic independence
was established.

The primary guarantee requires the complete fixed horizon, bets, roster, and decision thresholds to
be frozen before outcomes; no score-adaptive missingness, retry, replacement, or stopping; fresh
isolated provider requests and sandboxes for every arm; and score-blind whole-pair handling of
allowlisted infrastructure failures. Attempts inside a task may be dependent, but common provider
shocks, shared mutable state, overlapping sandboxes, correlated task scheduling incidents, or other
cross-task dependence invalidate the primary task-independence claim. Block arm order and equal
concurrency reduce drift but do not prove independence. Jackknife Student-t, Bonferroni jackknife,
and label swapping are model-based secondary diagnostics only and make no finite-sample alpha-control
claim.

For each provider lane, report:

- pi baseline score, candidate score, and paired percentage-point delta;
- the one-sided fixed-roster primary p-value and lower bound for every lane;
- the weighted semantic-cluster sensitivity p-value and lower bound for every lane, clearly labeled
  as a conservative sensitivity rather than a primary pass requirement;
- per-task win, tie, and loss counts;
- timeout and infrastructure rates;
- total input tokens, output tokens, provider cost, environment cost, and wall time;
- the post-timeout-verifier-reward sensitivity defined above.

Do not enable Harbor retries until the atomic attempt ledger records every failed and final arm,
including usage and exception type. Once that prerequisite exists, recover only explicitly
allowlisted infrastructure failures by replaying the whole pair under a new block-attempt identity,
and report every execution. The decision to replay must be score-blind. Never retry a task failure,
candidate failure, or low score.

The main success criteria are:

1. every lane exactly rejects its zero null, implying a positive one-sided primary bound on its
   fixed-roster conditional expected rerun delta, using unadjusted alpha 0.05 through the all-lane
   intersection-union rule;
2. every lane's observed equal-task delta is at least 3 percentage points; this is an effect-size
   floor, not by itself a powered 3 point claim;
3. the locked simulator's exact Monte Carlo gate passes at its predeclared MDE. A 3 point, 80% power
   statement is prohibited until a design with those values passes; otherwise report the larger
   predeclared powered MDE or state that power remains unresolved;
4. cost, timeout rate, and candidate-failure rate remain inside their predeclared noninferiority
   margins, with the score-cost decision rule and all margins frozen before search.

Do not make an inferential compatibility or equivalence claim against 76.4% from the mixed all-89
score. That score includes discovery task identities and outcomes used to select the harness, so
report only its signed descriptive distance from 76.4% and the external leaderboard context. State
any scoring-rule difference, including post-timeout reward treatment, alongside that distance. A
future independent suite or a predeclared selection-adjusted design is required for an inferential
equivalence statement. Claiming reproduction of the adaptive method additionally requires a future
separately funded all-89 lane to pass every initial-population, proposer, history, code-search,
schedule, and finalist-selection parity gate above. That lane is not part of this $15,000 study. An
exact serving-path reproduction remains out of scope while direct Anthropic is excluded.

The bootstrap ablation, cross-provider transfer, full-suite location relative to 76.4%, and any
future adaptive lane are secondary endpoints. Freeze a testing hierarchy or label them descriptive;
do not promote whichever secondary interval happens to exclude zero into a new primary claim.

## Budget and stop gates

The hard experiment budget is $15,000 across model calls, E2B, and recoverable reruns. Local compute
already available to the project is tracked but does not consume this ceiling.

| Envelope | Initial planning cap | Purpose |
|---|---:|---|
| qualification and backend parity | $500 | canaries, failure-path checks, local/E2B parity |
| matched baseline calibration | $2,000 | discovery pi baselines and optional control calibration |
| primary Bedrock search | $5,000 | up to ten frozen discovery iterations, capped by measured cost |
| Azure transfer evaluation | $1,000 | frozen Bedrock winner versus pi on the predeclared transfer matrix |
| locked final evaluations | $5,500 | paired pi/winner matrix, extending to all 89 tasks only if fully funded |
| audited retries and variance resolution | $500 | allowlisted infrastructure-only recovery |
| reserve | $500 | price drift, quota inefficiency, or completion of a predeclared blocked matrix |
| **total** | **$15,000** | hard ceiling |

These initial caps sum to the hard ceiling, but they are not yet proof that every listed cell is
funded. Phase 0 must freeze a finite, cost-only reallocation rule before qualification. After route
canaries, but before any qualification reward or trace is unsealed and before any Phase 2 baseline
or Phase 3 candidate is run, use only measured cost, quota, and infrastructure evidence to price the
exact confirmation split, the at-most-890-cell paired Bedrock final, the frozen Azure transfer
matrix, proposer calls, parity canaries, and any reference or Claude Code control. Apply the frozen
rule once, record the resulting phase caps and matrix, and verify that they sum to no more than
$15,000. Those revised caps are then immutable. No score, trace, or candidate outcome may trigger a
transfer between phases.

The confirmation cells are part of the 890-cell all-89 matrix, not an additional matrix. Reserve
the locked primary confirmation cells first. Extend the same final to discovery tasks only when all
890 cells fit. If the initially planned $5,500 final cap is insufficient, the predeclared rule may
move budget from search or transfer to the final before the immutable freeze. After that freeze,
omit the descriptive extension rather than weakening the paired primary comparison or reallocating
money after candidate selection.

No all-89 adaptive search is funded here. At the paper's planning estimate of roughly $500 for one
89-task, two-attempt iteration, its ten-iteration search alone would consume about $5,000 before a
separate 890-cell paired final. That lane needs a new measured budget and operator approval after
the primary study. Moving unused money into it mid-search would create an underpowered, shortened
lane that cannot carry a paper-method label.

The initial caps imply cumulative approval gates at $500, $2,500, $7,500, $8,500, $14,000, and
$15,000. When the one-time cost-only reallocation freezes revised phase caps, recompute the
cumulative gates as their running sums in execution order, retain $15,000 as the hard final stop,
and store both the caps and gates in the budget manifest. They are immutable from that point. At
each frozen gate, stop and record score, uncertainty, failures, actual cost per cell, remaining
matrix, and the value of the next spend. Discovery scores may support a predeclared primary-search
futility stop, but they may not expand the frozen proposal count or unlock a paper-method label.
Confirmation or Azure scores may never trigger a replacement run. Any predeclared sequential
extension must retain and analyze every valid attempt under a frozen combination rule; it may not
replace an inconvenient result. The reserve is not automatically available to search. No process
may start a paid phase without an explicit operator confirmation, and no job may exceed its
remaining envelope through automatic retry.

Every paid provider meter must be created through `provider_cost_meter()` before policy freeze. Its
registered offline verifier binds the exact nonsecret route, conservative price ceiling, retained
source digests, and interpreted record coordinates into the budget policy. Built-in Bedrock routes
use only the exact packaged evidence set registered for that route. Account-specific Azure routes
require explicit retained account, deployment, and bounded retail-price JSON; no verifier performs
ambient network access. The Azure evidence must bind the endpoint, deployment and ETag, immutable
model version, deployment SKU, region, retail service, meter IDs, units, effective date, and prices.
Record the evidence receipt digest in the sealed study manifest. Treat the receipt as a trusted
builder misconfiguration guard, not as a provider signature, and do not allow experiment input to
construct it directly.

## Current WMH readiness and remaining gaps

The reusable evaluation slice provides:

- Harbor as a pinned direct dependency and canonical task/verifier boundary;
- local and E2B Harbor environment configurations, with local as the default;
- a host-side Azure or Bedrock provider bridged to an isolated pi runner;
- immutable run manifests, task-lock digests, stale-run rejection, and strict result ingestion;
- an exclusive per-job resume lease and atomic replacement of Harbor's live root result;
- a benchmark-neutral `wmh harness eval` command and canonical result JSON;
- a synchronous `HarborHarnessScorer` that requires an exact literal task matrix, rejects compute
  drift, and projects only complete binary verifier evidence into generic harness scores;
- a crash-safe, cross-process provider spend ledger with immutable hard and phase caps,
  frozen route tariffs and estimators, conservative pre-call reservations, full-usage settlement,
  and full-audit plus incrementally verified worker state;
- typed completed, failed, unknown, and cancelled candidate evidence without arbitrary Harbor
  metadata passthrough;
- pre-job local enforcement of one literal prebuilt image with task Compose, dotenv, host
  interpolation, MCP/skills injection, run-level imports, mounts, overlays, and kwargs disabled;
- fail-closed rejection of unaudited Harbor retries;
- explicit separation of scored task failures from provider, environment, runner, and verifier
  infrastructure failures;
- an exact-uniform, score-independent sealed partition, fixed-roster paired primary report,
  weighted semantic-cluster sensitivity, and locked-simulation operating-characteristic gate.

The following remain required work before experiment launch:

- freeze and provenance-check the paper-target Terminal-Bench 2 and terminal-bench package commits;
- create the pi baseline and bootstrap ablation;
- generalize persistent execution, bounded output, multimodal observation, completion, caching,
  retry, and summarization surfaces enough to parity-test a reference-strength control;
- add the proposer and Pareto search loop on top of existing harness deltas, including a sealed
  discovery-only workspace, immutable proposer identity, complete-history evidence, the matched-
  compute eligibility gate, deterministic winner selection, and cost evidence;
- implement an atomic attempt ledger that retains every attempt's usage, failure type, and terminal
  result before enabling retry or search ranking;
- implement an external-resource ledger that reconciles local Compose and E2B resource creation and
  teardown after process or host interruption;
- supply and lock the actual paired simulation design, pass its predeclared MDE gate, and add the
  fixed deterministic action-replay canary manifest and blocked two-arm final scheduler;
- add served-model attestation to canonical evidence, including the response-reported Azure model
  revision and a pre/post deployment snapshot; the current run digest binds the declared endpoint,
  deployment, model, API version, and Bedrock region, but cannot detect an Azure deployment repoint;
- bind the evaluator worker build or WMH source revision into portable evidence; the current digest
  relies on a manually managed agent version, so Phase 0 must separately freeze a clean commit;
- bind every task image to a resolved content digest and isolate the evaluator against Harbor's
  fully buffered command-output limitation;
- bind the resolved pi-runner platform and child-manifest digest after cross-platform qualification;
- make Harbor's remaining root and per-trial config, lock, and result writes crash-safe, or validate
  a fail-closed snapshot recovery procedure;
- add deadline-aware, interruptible provider calls, prove provider and Harbor subprocess cleanup,
  freeze the timeout stack, and probe the Azure/Bedrock failure taxonomy;
- classify candidate-authored request, context, and tool-schema 4xx failures as candidate zero
  while retaining authentication, routing, transport, throttling, and service failures as
  infrastructure; otherwise one invalid proposal can veto the optimizer;
- classify candidate-caused task-container destruction or resource exhaustion as candidate zero,
  with separate run-health evidence for ambiguous failures, instead of treating every task-tool
  transport exception as infrastructure;
- make remote, registry, and package task acquisition preserve symlinks for pre-read validation,
  or keep all ground-truth evaluation restricted to preflighted local dataset paths;
- add and fund a generic Claude Code control only if making a matched headline-uplift claim;
- perform real local/E2B parity canaries on a machine with Docker and valid E2B credentials;
- run leakage audits and the paid Azure/Bedrock matrices.

These gaps must close before a paid experiment run begins. They should be implemented as generic
search, budgeting, and analysis components driven by experiment configuration, not as paper-named
branches in the WMH core.

As a compatibility check on July 18, 2026, the official Terminal-Bench 2 repository at the pinned
paper-target commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` contained 89 task documents. All
89 passed Harbor 0.18 parsing, the strict local prebuilt-image policy, and the no-follow immutable
dataset snapshot path; none uses a separate verifier environment. This source qualification does
not replace freezing each resolved trial lock, task checksum, executed image digest/platform, or
the pinned terminal-bench package commit in the paid-run manifest.

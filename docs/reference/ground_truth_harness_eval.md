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
generalization. The primary WMH lane instead freezes family-stratified discovery and confirmation
partitions before search. It scores candidates only on discovery tasks, selects one winner, and
evaluates that winner once on the confirmation partition sealed from experiment-time search
artifacts and rewards. The empirical matched delta on that realized confirmation matrix is the
assumption-free primary observation. Interpreting it as generalization to a task population requires
the exchangeability, sampling, and contamination assumptions defined in the power gate below.

After selection, the primary winner may also be compared descriptively with pi on all 89 tasks at
five trials per task. That full-suite score is useful for locating the candidate relative to 76.4%,
but it is not a reproduction of the paper's adaptive all-89 method because confirmation tasks did
not participate in selecting that candidate.

A separate all-89 adaptive lane would require its own search over all 89 tasks, separately selected
candidate, and locked five-trial all-89 final. It is deferred from the $15,000 protocol below. A
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
task IDs, family strata, random seed, and powered minimum detectable effect before the first
candidate-scoring call.

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
| at least 1 and less than 3 points | promising; conditional CI must exclude zero for a population claim |
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

- **local**, the default, for Docker-backed development, deterministic debugging, and final runs
  when capacity permits;
- **e2b**, an explicit acceleration option for high-concurrency task execution.

These choices move only Harbor's task environment. The isolated pi runner remains in local Docker
for both, so the evaluator host requires Docker even when `--task-backend e2b` is selected.

Before any scored E2B run, replay one frozen scripted or golden action sequence against identical
task locks on local and E2B. Compare normalized command outcomes, verifier rewards, final task
state, timeout classification, and environment fingerprints. This deterministic replay, not exact
reward equality from a stochastic model run, is the backend parity gate. If live pi samples are
also used to claim backend equivalence, freeze a separate paired design, equivalence margin,
sample size, and task-clustered analysis with adequate power. Otherwise report those live samples
as descriptive route evidence only. Provider calls remain on the trusted host in both modes.
Provider credentials must not enter task containers, E2B sandboxes, candidate prompts, traces, or
canonical result JSON. Before Harbor
constructs a task environment, WMH audits each resolved task source and rejects credential-like
host variable references in task and verifier environment maps, Docker Compose sources, and
Compose environment files. The audit reports variable names and source locations only. It never
resolves or logs credential values.

### Python scoring boundary

`HarborHarnessScorer` is the synchronous bridge from one explicit Harbor task selection to the
generic harness-search score contract. The dataset selection and `task_ids` must be the same
ordered, literal list. Globs, exclusions, random task limits, request-time subsets, and
request-time attempt changes are rejected. Scored search accepts only local dataset paths that WMH
can inspect before Harbor job creation. Harbor 0.18 dereferences symlinks while copying remote/Git
tasks, before WMH can validate the downloaded tree, so registry, package, and remote acquisition
remain evaluator-only until a symlink-preserving acquisition boundary lands. Dataset-qualified task
keys from a frozen qualification
manifest bind each selection to its Harbor source, identity, and task checksum across candidates.
Qualification also freezes the executed environment definition. Local runs attest the Docker
daemon platform plus every Compose service's immutable image ID and image platform. Ephemeral
container, trial, and project identities are deliberately excluded. Every scored attempt must
reproduce the per-task qualification digest.

Harbor 0.18 does not expose the immutable E2B build ID that `Sandbox.create` actually resolved.
Looking up the mutable `default` tag after sandbox creation cannot close the alias race, even when
the current tag and sandbox timestamps are retained. The evaluator therefore keeps E2B selectable
for acceleration and parity diagnostics, but `HarborHarnessScorer` rejects E2B scored search until
the integration creates sandboxes from an immutable build reference and surfaces that identity.

```python
from harbor.models.job.config import DatasetConfig

from wmh.evals.harbor.config import HarborJobSpec
from wmh.evals.harbor.scorer import HarborHarnessScorer
from wmh.harness.scoring import ScoreRequest

task_ids = ("task-a", "task-b")
# Dataset-qualified task keys come from the frozen Harbor qualification manifest.
qualified_task_keys = {
    trial.task_identity: trial.cell.task_key for trial in qualification.result.trials
}
task_keys = tuple(qualified_task_keys[task_id] for task_id in task_ids)
qualified_environment_digests = {}
for task_id in task_ids:
    digests = {
        trial.task_environment_digest
        for trial in qualification.result.trials
        if trial.task_identity == task_id
    }
    if None in digests or len(digests) != 1:
        raise ValueError(f"qualification did not freeze one environment for {task_id!r}")
    qualified_environment_digests[task_id] = digests.pop()
task_environment_digests = tuple(
    qualified_environment_digests[task_id] for task_id in task_ids
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
    reference_harness=baseline,
    task_ids=task_ids,
    task_keys=task_keys,
    task_environment_digests=task_environment_digests,
    reward_key="reward",
) as scorer:
    report = scorer.score(candidate, request=ScoreRequest(purpose="full"))
```

When this scorer is passed to `search_harness`, set `screen_proposals=False` and
`confirm_narrow_vetoes=False`. Every candidate then receives exactly the frozen matrix. The
reference harness fixes runtime kind, turns, output tokens, temperature, effective tools, and the
per-turn deadline. A candidate that changes that compute envelope is ineligible.

This static check is a narrow credential boundary, not proof that an arbitrary Harbor task is safe.
A malicious Dockerfile, Compose mount, image, script, or verifier can attack the host through other
channels. Every scored dataset must therefore be frozen by content, reviewed as executable code,
and limited to audited task sources and image digests before credentials are loaded or a paid run
is approved.

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
a Harbor environment failure remains infrastructure; exhaustion of the caller-owned turn deadline
remains a gradeable candidate timeout. These controls handle ordinary commands and frozen task
images. They are not a trusted kill boundary against a hostile root candidate: root can replace the
task-side shell or cap utilities, detach descendants, or otherwise defeat a mutable in-container
supervisor, while Harbor exposes no portable descendant-kill proof.

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

The same interface exposes raw provider exceptions without a portable failure class. Azure and
Bedrock do not use one common signal for context overflow, policy intervention, authentication,
quota, and transient service failure. WMH currently classifies an exception conservatively as
provider infrastructure and invalidates the cell. `HarborJobSpec` rejects every nonzero retry
configuration because Harbor 0.18 overwrites the final trial result without retaining complete
attempt usage and exceptions. Before enabling paid retry or search ranking, add an atomic attempt
ledger and a typed adapter-level taxonomy validated by live probes on the frozen Azure and Bedrock
routes. Only exact deterministic context or policy signals may become gradeable candidate outcomes;
ambiguous errors must stay infrastructure failures.

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
candidate calls are admitted. Keep the split seed, confirmation IDs, and family assignment in a
separate operator-owned location that is never mounted into that workspace.

Family metadata used by the split generator must have frozen provenance and must be independent of
WMH, candidate, and task-level leaderboard rewards. Only the sealed control plane assigns tasks to
strata and partitions. Aggregate discovery stratum counts may be published to the proposer when
needed, but confirmation membership may not. Public availability and possible model pretraining on
Terminal-Bench remain limitations. This seal establishes only that search receives no confirmation
artifacts or rewards through experiment-time storage, tools, or network access. It cannot show that
confirmation content was absent from model training, prevent a proposer with a memorized public task
roster from inferring the complement of the discovery set, or rule out consequential pretraining
contamination. A task-population generalization interpretation therefore requires the registered
assumption that retained training information neither lets search target confirmation content nor
makes confirmation performance unrepresentative of the declared population. If that assumption is
not defensible, the confirmation result remains a descriptive matched delta on the realized sealed
matrix only. The seal cannot verify this assumption; every population claim must label it explicitly
and accompany it with the descriptive result.

The final evaluator may of course expose one confirmation task instruction and environment to the
frozen worker candidate while that task is running. It must not expose confirmation content to the
search proposer or return final evidence until candidate identity is frozen. The post-run string
and source audit is defense in depth, not a replacement for this access boundary.

#### Power and partition gate

Before any benchmark reward or candidate score is observed, write and freeze a power-design
manifest. It must predeclare the baseline pass rate, within-task intraclass correlation, any paired
arm correlation used by the data-generating model, the target effect, test direction, alpha, power,
attempt count, family strata, stratum population sizes, task inclusion probabilities, estimator,
confidence-interval algorithm, bootstrap resample count and seed, and minimum discovery-set
constraints. It must also state the inferential target and the assumptions that make it
identifiable. Within each frozen family stratum, benchmark tasks must be exchangeable draws from the
declared Terminal-Bench-regime task superpopulation, and a future task must be another such draw. The
score-independent split must be independent of task potential outcomes, and information retained in
proposer or worker model weights must neither let search target confirmation membership or content
nor make confirmation outcomes unrepresentative of the declared population.

Conditional on those assumptions and on the realized discovery search, the primary inferential
estimand is the expected paired reward delta of the frozen winner versus pi on a fresh task from
that task population, standardized to the frozen family shares of the 89-task benchmark. The primary
empirical statistic is the corresponding family-standardized paired delta on the realized
confirmation matrix. Also report the unweighted, task-uniform paired delta across the realized
confirmation tasks and planned attempts. Both empirical statistics are assumption-free descriptions
of that realized matrix. Because the candidate is selected as a function of the discovery
complement, neither confirmation statistic identifies the selected candidate's finite-population
mean over all 89 tasks. Use a 3 percentage-point paired effect as the target, a two-sided alpha of
0.05, power of at least 0.80, and five attempts per task for the locked primary confirmation. Derive
nuisance assumptions from published or otherwise external evidence, and run sensitivity cases over
a predeclared plausible range rather than fitting them to WMH candidate outcomes.

Use task-clustered simulation that generates complete tasks with attempts nested inside each task,
applies the exact planned missing-cell rule, and runs the same paired analysis that will produce the
final interval. Sample tasks within each frozen family stratum. Estimate the family-standardized
task-population mean with inverse-inclusion weights, which is equivalent to weighting each
confirmation stratum mean by its frozen share of the 89 tasks under fixed within-stratum sample
counts. Resample complete tasks only within their strata for the bootstrap. Throughout the nuisance
range and within a predeclared Monte Carlo tolerance, the simulation must verify at least 0.80 power
at the target effect, empirical two-sided type-I error no greater than 0.05, and interval coverage
of at least 0.95. Those operating characteristics are conditional on the registered task-population
and data-generating assumptions; they are not design-based inference for the selected candidate's
finite 89-task mean.

Enumerate integer, family-stratified discovery and confirmation allocations across the 89 frozen
tasks. Select task counts before scoring: choose a confirmation count that passes every power,
type-I, and coverage gate while preserving the declared minimum discovery matrix, then assign task
IDs using only frozen, score-independent family metadata and the predeclared seed. Record the
simulation code, inputs, output, selected task counts, inclusion probabilities, analysis seed, task
IDs, and matrix digest in the manifest.

The same manifest must freeze the Azure transfer task and attempt matrix, or a finite conditional
ladder of matrices selected only by measured per-cell cost and quota. If a ladder is used, record
its thresholds before qualification, keep candidate rewards hidden, and resolve it from cost and
infrastructure evidence only before Bedrock search begins. Azure scores must never choose the
transfer matrix.

If no valid allocation of the 89 tasks reaches 0.80 power for a 3 point effect, the 3 point
confirmation gate fails before spending. Choose and record exactly one route before scoring: stop
and add independent compatible tasks under a separately defined broader estimand; retain a held-out
confirmation lane with the larger minimum detectable effect supported by the same simulation and
treat 3 points as descriptive only; or use a locked all-89 paired estimate as a
benchmark-descriptive endpoint and abandon the task-population generalization claim. Do not lower
the power target, choose the route after seeing scores, or present an underpowered interval as
confirmation of a 3 point effect.

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

- Primary confirmation: pi baseline and the primary winner on the frozen confirmation partition,
  five attempts per task.
- Full-suite descriptive comparison: if it is fully funded, extend that same blocked final matrix
  to the discovery tasks so pi and the frozen primary winner each receive five attempts on all 89
  tasks with the fixed primary model. Compute the primary endpoint from only the prespecified
  confirmation subset and the descriptive score from all 89 tasks. Do not rerun confirmation cells
  in a second full-suite job. This comparison is not an adaptive-method reproduction. Add the
  reference-strength control only if the final-evaluation envelope still covers every paired
  primary cell. If all 89 tasks do not fit, run the confirmation matrix and omit the full-suite
  descriptive score.
- All-89 adaptive comparison: deferred and unfunded in this protocol. It requires a new budget that
  covers the complete frozen search plus a separate five-attempt paired final before its first
  candidate call.
- Transfer lane: the already-frozen Bedrock winner and its matching pi baseline on the frozen Azure
  deployment. Before search, bind the exact task IDs and attempt count to a score-independent
  power-and-cost design and its matrix digest. Use at least three attempts per selected task, and do
  not change the matrix after observing any Azure candidate comparison. Use the same confirmation
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

For a cell that finishes without a candidate-owned task timeout, its analysis outcome is the
selected binary verifier reward. A candidate-owned agent or task timeout has primary outcome zero,
even if Harbor later obtains a valid verifier reward from the final task state. Preserve that reward
and the timeout failure kind as diagnostic evidence. A verifier, provider, environment, runner, or
unclassified infrastructure failure is not a zero and invalidates the planned matrix until an
allowlisted, fully ledgered recovery fills that cell. The timeout sensitivity analysis substitutes
Harbor's valid post-timeout verifier reward for the primary zero; it may not replace the primary
result after scores are observed.

First average the planned analysis outcomes within each task and arm: five for primary confirmation
and the separately frozen count for Azure transfer. The primary empirical metric is the paired
difference of inverse-inclusion-weighted task means over the exact confirmation matrix, standardized
to the frozen family shares of the 89-task benchmark. Under the registered exchangeability,
sampling, and contamination assumptions, it estimates the conditional task-population
generalization delta of the frozen winner versus pi. Use frozen family shares and resample complete
paired tasks within family strata using the predeclared bootstrap algorithm, resample count, and
seed. The bootstrap interval has that population interpretation only under those assumptions. Also
report the unweighted task-uniform matched delta on the realized confirmation matrix as an
additional empirical quantity. Both point estimates are assumption-free descriptions of the
realized matrix; neither confirmation metric nor its interval identifies the selected candidate's
finite all-89 mean. The metric is valid only if every planned cell has a primary analysis outcome
and the frozen inclusion weight.

For each provider lane, report:

- pi baseline score, candidate score, and paired percentage-point delta;
- the frozen family-stratified task bootstrap confidence interval over the paired delta, labeled as
  conditional on the registered task-population assumptions;
- per-task win, tie, and loss counts;
- timeout and infrastructure rates;
- total input tokens, output tokens, provider cost, environment cost, and wall time;
- the post-timeout-verifier-reward sensitivity defined above.

Do not enable Harbor retries until the atomic attempt ledger records every failed and final attempt,
including usage and exception type. Once that prerequisite exists, retry only explicitly
allowlisted infrastructure failures, reuse the same task identity with a distinct attempt identity,
and report every attempt. Never retry a task failure, candidate failure, or low score.

The main success criteria are conditional on passing the power and partition gate. Criteria 1 and 2
support task-population generalization only while the registered exchangeability, sampling, and
contamination assumptions remain defensible. If an audit breaks one of those assumptions, report the
matched confirmation delta as descriptive and do not call it confirmed generalization:

1. a positive family-standardized paired delta over pi whose conditional task-population confidence
   interval excludes zero on the confirmation partition sealed from experiment-time search evidence
   in the primary provider lane;
2. at least a 3 point conditional task-population improvement when the frozen design has at least
   0.80 power for that effect, or the larger predeclared minimum detectable effect selected by the
   fail route; a smaller improvement that holds across both providers remains useful but is not a
   powered 3 point result;
3. cost, timeout rate, and candidate-failure rate remain inside their predeclared noninferiority
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
- typed completed, failed, unknown, and cancelled candidate evidence without arbitrary Harbor
  metadata passthrough;
- pre-environment rejection of task-authored imports of credential-like host variables;
- fail-closed rejection of unaudited Harbor retries;
- explicit separation of scored task failures from provider, environment, runner, and verifier
  infrastructure failures.

The following remain required work before experiment launch:

- freeze and provenance-check the paper-target Terminal-Bench 2 and terminal-bench package commits;
- create the pi baseline and bootstrap ablation;
- generalize persistent execution, bounded output, multimodal observation, completion, caching,
  retry, and summarization surfaces enough to parity-test a reference-strength control;
- add the proposer and Pareto search loop on top of existing harness deltas, including a sealed
  discovery-only workspace, immutable proposer identity, complete-history evidence, the matched-
  compute eligibility gate, deterministic winner selection, and cost evidence;
- implement a persistent spend ledger and phase-budget admission check;
- implement an atomic attempt ledger that retains every attempt's usage, failure type, and terminal
  result before enabling retry or search ranking;
- implement an external-resource ledger that reconciles local Compose and E2B resource creation and
  teardown after process or host interruption;
- implement the paired statistical report, powered and immutable split generator, fixed
  deterministic action-replay canary manifest, and blocked two-arm final scheduler;
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
- implement a fail-closed local Compose host policy from compatibility data, with an audited
  allowlist for bind mounts and build contexts and unconditional rejection of privileged mode,
  host devices, Docker socket access, build SSH forwarding, and any `HOST_*_PATH` or equivalent
  evidence-path escape; apply it before provider or E2B credentials are loaded and before any paid
  local task starts;
- add deadline-aware, interruptible provider calls, prove provider and Harbor subprocess cleanup,
  freeze the timeout stack, and probe the Azure/Bedrock failure taxonomy;
- classify candidate-authored request, context, and tool-schema 4xx failures as candidate zero
  while retaining authentication, routing, transport, throttling, and service failures as
  infrastructure; otherwise one invalid proposal can veto the optimizer;
- classify candidate-caused task-container destruction or resource exhaustion as candidate zero,
  with separate run-health evidence for ambiguous failures, instead of treating every task-tool
  transport exception as infrastructure;
- create E2B sandboxes from an immutable build reference and surface the exact resolved build ID
  before admitting E2B scores; keep E2B acceleration diagnostic-only until that boundary lands;
- make remote, registry, and package task acquisition preserve symlinks for pre-read validation,
  or keep scored search restricted to preflighted local dataset paths;
- add and fund a generic Claude Code control only if making a matched headline-uplift claim;
- perform real local/E2B parity canaries on a machine with Docker and valid E2B credentials;
- run leakage audits and the paid Azure/Bedrock matrices.

These gaps must close before a paid experiment run begins. They should be implemented as generic
search, budgeting, and analysis components driven by experiment configuration, not as paper-named
branches in the WMH core.

As a compatibility check, the official Terminal-Bench 2 repository at commit
`2fd12b88aafdd04a52c298e3940bcb189f9766d6` contained 89 task documents on July 18, 2026. All 89
parsed under Harbor 0.18 without a separate verifier environment, so the evaluator's fail-closed
separate-verifier restriction does not block that snapshot. This repository inspection does not
establish compatibility with the paper-target commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`
or its pinned terminal-bench package. It also does not replace freezing the Harbor registry
resolution and per-task checksums used by the paid run.

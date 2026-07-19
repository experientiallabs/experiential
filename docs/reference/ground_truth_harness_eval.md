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

The headline improvement is 18.4 percentage points over Claude Code. The difference from the
official Terminus-KIRA leaderboard comparator is only 1.7 points. Both comparisons matter. A result
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
generalization. The primary WMH lane instead freezes task-family-balanced discovery and
confirmation partitions before search. It scores candidates only on discovery tasks, selects one
winner, and evaluates that winner once on the untouched confirmation partition.

After selection, the primary winner may also be compared descriptively with pi on all 89 tasks at
five trials per task. That full-suite score is useful for locating the candidate relative to 76.4%,
but it is not a reproduction of the paper's adaptive all-89 method because confirmation tasks did
not participate in selecting that candidate.

An optional all-89 adaptive lane requires a separate search over all 89 tasks and a separately
selected candidate, followed by its own locked five-trial all-89 final. It qualifies as a
paper-method lane only if it starts from a parity-passed reference-strength seed corresponding to
the released KIRA-derived scaffold and its locked final includes that same seed as the control. If
that seed is unavailable, the lane may proceed only as a generic all-89 adaptive study and must not
be described as reproducing the paper's method. Its scores, proposer history, candidate identity,
and budget ledger must remain distinct from the confirmation-valid primary lane. Confidence
intervals from this optional lane describe repeated-trial variability after selection on the same
tasks; they do not establish generalization to unseen tasks. Freeze the primary lane's task IDs,
family strata, random seed, and powered minimum detectable effect before the first
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
budget, trial concurrency, agent concurrency, and provider retry policy across arms. The run
configuration digest binds both concurrency controls because throttling and contention can change
timeouts and scores. If a candidate intentionally changes one of those
resources, report it on a separate score-cost Pareto endpoint rather than attributing the entire
gain to harness quality at matched compute.

The most likely early improvement is a compact environment bootstrap before the first agent turn.
The paper's winning harness primarily added an initial snapshot of the working directory, `/app`,
installed languages, tools, package managers, and memory. The paper reports two to four saved
exploration turns, while the released artifact README says two to five. This is a hypothesis to
test, not a feature to hard-code.

Expected outcomes, in decreasing order of confidence:

- The optimized candidate beats stock WMH pi on the matched Bedrock lane.
- The environment-bootstrap ablation accounts for a material share of the improvement.
- A useful fraction of the gain transfers to the Azure lane, but the absolute score differs because
  Azure and Bedrock do not provide the same model family.
- Matching 76.4% from a stock pi seed is possible but not the planning assumption. The winning
  harness builds on Terminus-KIRA. The paper's documented search run evaluates its KIRA baseline at
  64.4%, while 74.7% is the separate official-leaderboard KIRA comparator used in the final table.
  A result near the headline likely requires adapting that seed or independently recovering
  comparable capabilities first.

Before seeing final results, use these interpretation bands:

| Matched improvement over stock pi | Interpretation |
|---|---|
| less than 1 point | no actionable improvement |
| 1 to 3 points | promising, but require a confidence interval excluding zero |
| 3 to 5 points | practically meaningful harness improvement |
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
  model-matched reproduction and search lane. Confirm account access and quota before freezing the
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

Before any scored E2B run, run a backend parity canary with the same candidate and task locks.
Compare verifier rewards, final task state, timeout classification, and environment fingerprints.
Provider calls remain on the trusted host in both modes. Provider credentials must not enter task
containers, E2B sandboxes, candidate prompts, traces, or canonical result JSON. Before Harbor
constructs a task environment, WMH audits each resolved task source and rejects credential-like
host variable references in task and verifier environment maps, Docker Compose sources, and
Compose environment files. The audit reports variable names and source locations only. It never
resolves or logs credential values.

This static check is a narrow credential boundary, not proof that an arbitrary Harbor task is safe.
A malicious Dockerfile, Compose mount, image, script, or verifier can attack the host through other
channels. Every scored dataset must therefore be frozen by content, reviewed as executable code,
and limited to audited task sources and image digests before credentials are loaded or a paid run
is approved.

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

Current agent evidence records input and output tokens, but not cache-token detail or authoritative
provider cost. The canonical result therefore keeps `cache_tokens` and `cost_usd` missing rather
than estimating them from a static price table. The spend ledger must reconcile provider usage and
contracted billing outside the task environment before it admits the next paid block.

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
6. Register the $15,000 hard ceiling and per-phase stop limits.

#### Power and partition gate

Before any benchmark reward or candidate score is observed, write and freeze a power-design
manifest. It must predeclare the baseline pass rate, within-task intraclass correlation, any paired
arm correlation used by the data-generating model, the target effect, test direction, alpha, power,
attempt count, family strata, and minimum discovery-set constraints. Use a 3 percentage-point
paired effect as the target, a two-sided alpha of 0.05, power of at least 0.80, and five attempts per
task for the locked primary confirmation. Derive nuisance assumptions from published or otherwise
external evidence, and run sensitivity cases over a predeclared plausible range rather than fitting
them to WMH candidate outcomes.

Use task-clustered simulation that generates complete tasks with attempts nested inside each task,
applies the exact planned missing-cell rule, and runs the same task-clustered paired analysis that
will produce the final interval. Enumerate integer, family-balanced discovery and confirmation
allocations across the 89 frozen tasks. Select task counts before scoring: choose a confirmation
count that reaches at least 0.80 power throughout the predeclared nuisance range while preserving
the declared minimum discovery matrix, then assign task IDs using only frozen, score-independent
family metadata and the predeclared seed. Record the simulation code, inputs, output, selected task
counts, task IDs, and matrix digest in the manifest.

The same manifest must freeze the Azure transfer task and attempt matrix, or a finite conditional
ladder of matrices selected only by measured per-cell cost and quota. If a ladder is used, record
its thresholds before qualification, keep candidate rewards hidden, and resolve it from cost and
infrastructure evidence only before Bedrock search begins. Azure scores must never choose the
transfer matrix.

If no valid allocation of the 89 tasks reaches 0.80 power for a 3 point effect, the 3 point
confirmation gate fails before spending. Choose and record exactly one route before scoring: stop
and add independent compatible tasks; retain a held-out confirmation lane with the larger minimum
detectable effect supported by the same simulation and treat 3 points as descriptive only; or use a
locked all-89 paired estimate as a benchmark-descriptive endpoint and abandon the unseen-task
generalization claim. Do not lower the power target, choose the route after seeing scores, or present
an underpowered interval as confirmation of a 3 point effect.

### Phase 1: zero-cost and low-cost qualification

Run schema, manifest, resume, candidate-failure, timeout, verifier, and credential-isolation tests.
Then run a frozen cross-family task canary with one attempt per task:

- pi baseline on local;
- pi baseline on E2B;
- bootstrap ablation on local;
- one Azure and one Bedrock route.

Freeze the task IDs and an absolute allowed-failure count before scoring. Stop if canonical results
contain unclassified failures, if local and E2B disagree on successful task state, or if
infrastructure failures exceed that count after allowlisted retries. Report the percentage too, but
do not use a percentage alone as the stopping rule for a small sample.

Use pi, not a mutable candidate, for the Azure route canary. If its cost or quota evidence resolves
a predeclared transfer-matrix ladder, quarantine its reward until the matrix and primary Bedrock
winner are frozen.

### Phase 2: establish matched baselines

Run the stock pi baseline on the frozen discovery matrix with two attempts per task in each provider
lane. Run the reference-strength control in the primary Bedrock lane only after the runtime and tool
parity gate passes, and require it if the study will make a paper-result claim. Freeze these results
before search. Do not run or reveal baseline scores, traces, or verifier outcomes on confirmation
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

Run candidates concurrently through Harbor, subject to provider quotas and the dollar gate. Do not
run multiple statistical attempts for the same task concurrently if provider or environment rate
limits could create candidate-dependent throttling.

The official released search configuration uses all 89 tasks, two trials per task, concurrency 50,
and reports roughly $500 and four to six hours per iteration with Opus 4.6. Treat those figures as a
planning reference only. Recalculate from live Azure, Bedrock, Harbor, and E2B prices before launch.

At the primary search limit, select and freeze its winner without opening confirmation evidence. If
the optional all-89 adaptive lane is activated, start it only after that freeze, use a separate
search history from a declared seed, score its candidates on all 89 tasks, and select its winner
independently. Call it a paper-method lane only when the seed is the parity-passed
reference-strength control corresponding to the released scaffold. Otherwise label it a generic
all-89 adaptive lane and prohibit a paper-method reproduction claim. The primary winner cannot be
relabeled as the all-89 winner, even if it later receives an all-89 descriptive score.

The primary winner must be frozen before any Azure candidate comparison used for the transfer
endpoint. Do not select, revise, or rank it using Azure results. An Azure-guided iteration belongs in
a separately budgeted exploratory lane and cannot support the cross-provider transfer claim.

### Phase 4: locked confirmation

Each lane's winner must already be frozen. Make no further harness changes after observing its
locked results.

- Primary confirmation: pi baseline and the primary winner on the frozen confirmation partition,
  five attempts per task.
- Full-suite descriptive comparison: pi baseline and the same primary winner on all 89 tasks, five
  attempts per task, using the fixed primary model. This comparison is not an adaptive-method
  reproduction. Add the reference-strength control only if the final-evaluation envelope still
  covers every paired primary cell.
- Optional all-89 adaptive comparison: the lane's frozen seed/control and its separately selected
  winner on all 89 tasks, five attempts per task. It is a paper-method comparison only when that
  seed is the parity-passed reference-strength control corresponding to the released scaffold.
  Otherwise report it as generic all-89 adaptive search. Add pi as another locked control only if
  every resulting pair is fully funded. Omit the lane unless its search and paired final are fully
  funded before the first candidate call.
- Transfer lane: the already-frozen Bedrock winner and its matching pi baseline on the frozen Azure
  deployment. Before search, bind the exact task IDs and attempt count to a score-independent
  power-and-cost design and its matrix digest. Use at least three attempts per selected task, and do
  not change the matrix after observing any Azure candidate comparison.

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

The primary metric is the arithmetic mean of the selected binary reward over the exact planned
matrix. A mean is valid only if every planned cell is scored and carries that reward key.

For each provider lane, report:

- pi baseline score, candidate score, and paired percentage-point delta;
- a task-clustered bootstrap confidence interval over the paired delta;
- per-task win, tie, and loss counts;
- timeout and infrastructure rates;
- total input tokens, output tokens, provider cost, environment cost, and wall time;
- sensitivity to treating valid verifier scores after task timeout as scored timeouts.

Do not enable Harbor retries until the atomic attempt ledger records every failed and final attempt,
including usage and exception type. Once that prerequisite exists, retry only explicitly
allowlisted infrastructure failures, reuse the same task identity with a distinct attempt identity,
and report every attempt. Never retry a task failure, candidate failure, or low score.

The main success criteria are conditional on passing the power and partition gate:

1. a positive paired delta over pi whose confidence interval excludes zero on the untouched
   confirmation partition in the primary provider lane;
2. at least a 3 point practical improvement when the frozen design has at least 0.80 power for that
   effect, or the larger predeclared minimum detectable effect selected by the fail route; a smaller
   improvement that holds across both providers remains useful but is not a powered 3 point result;
3. no material regression in cost, timeout rate, or candidate-failure rate that erases the score
   benefit;
4. for a primary candidate described as statistically compatible with 76.4%, a locked all-89
   descriptive run whose point estimate and uncertainty meet a predeclared equivalence criterion
   centered on 76.4%, and whose matched controls behave consistently with the published ordering.
   Merely producing a wide confidence interval that includes 76.4% is not evidence of reproduction.
   Claiming reproduction of the adaptive method additionally requires the separately searched and
   selected optional all-89 lane to use the parity-passed reference-strength seed and locked
   control. An exact serving-path reproduction remains out of scope while direct Anthropic is
   excluded.

## Budget and stop gates

The hard experiment budget is $15,000 across model calls, E2B, and recoverable reruns. Local compute
already available to the project is tracked but does not consume this ceiling.

| Envelope | Maximum | Purpose |
|---|---:|---|
| qualification and backend parity | $500 | canaries, failure-path checks, local/E2B parity |
| matched baseline calibration | $2,000 | discovery pi baselines and optional control calibration |
| primary Bedrock search | $5,000 | up to ten paper-sized iterations, capped by measured cost |
| Azure transfer evaluation | $1,000 | frozen Bedrock winner versus pi on the predeclared transfer matrix |
| locked final evaluations | $4,000 | paired pi/winner matrices, plus control only if fully funded |
| optional all-89 adaptive search | $1,500 | separate reference-seeded or generic candidate, only if the predeclared minimum search fits |
| audited retries and variance resolution | $500 | allowlisted infrastructure-only recovery |
| reserve | $500 | price drift, quota inefficiency, or one predeclared decisive rerun |
| **total** | **$15,000** | hard ceiling |

These envelopes sum to the hard ceiling, but they are not yet proof that every listed cell is
funded. Price the exact confirmation split, 890 paired Bedrock full-suite cells, the frozen Azure
transfer matrix, proposer calls, parity canaries, and any reference or Claude Code control from
measured canary cost before search. The optional all-89 envelope is reserved but not automatically
spendable: activate it only if measured cost shows that its predeclared minimum number of search
iterations and its separate paired final both fit. Otherwise omit that lane or explicitly reallocate
funds before either search
begins. Reserve the locked primary final matrix first. If the $4,000 final envelope is insufficient,
reduce search or transfer scope before spending rather than weakening the paired primary comparison
after candidate selection.

Use cumulative approval gates at $500, $2,500, $7,500, $12,500, and $15,000. At each gate, stop and
record score, uncertainty, failures, actual cost per cell, remaining matrix, and the value of the
next spend. The reserve is not automatically available to search. No process may start a paid phase
without an explicit operator confirmation, and no job may exceed its remaining envelope through
automatic retry.

## Current WMH readiness and remaining gaps

The reusable evaluation slice provides:

- Harbor as a pinned direct dependency and canonical task/verifier boundary;
- local and E2B Harbor environment configurations, with local as the default;
- a host-side Azure or Bedrock provider bridged to an isolated pi runner;
- immutable run manifests, task-lock digests, stale-run rejection, and strict result ingestion;
- an exclusive per-job resume lease and atomic replacement of Harbor's live root result;
- a benchmark-neutral `wmh harness eval` command and canonical result JSON;
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
- add the proposer and Pareto search loop on top of existing harness deltas, including immutable
  proposer identity and cost evidence;
- implement a persistent spend ledger and phase-budget admission check;
- implement the paired statistical report, powered and immutable split generator, fixed canary
  manifest, and blocked two-arm final scheduler;
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
- add deadline-aware provider calls, an explicitly frozen paper-strength timeout stack, and a
  probed Azure/Bedrock failure taxonomy;
- add and fund a generic Claude Code control only if making a matched headline-uplift claim;
- perform real local/E2B parity canaries on a machine with Docker and valid E2B credentials;
- run leakage audits and the paid Azure/Bedrock matrices.

These gaps must close before a paid reproduction run begins. They should be implemented as generic
search, budgeting, and analysis components driven by experiment configuration, not as paper-named
branches in the WMH core.

As a compatibility check, the official Terminal-Bench 2 repository at commit
`2fd12b88aafdd04a52c298e3940bcb189f9766d6` contained 89 task documents on July 18, 2026. All 89
parsed under Harbor 0.18 without a separate verifier environment, so the evaluator's fail-closed
separate-verifier restriction does not block that snapshot. This repository inspection does not
establish compatibility with the paper-target commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`
or its pinned terminal-bench package. It also does not replace freezing the Harbor registry
resolution and per-task checksums used by the paid run.

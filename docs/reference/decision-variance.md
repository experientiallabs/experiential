# Decision-variance environments

This pipeline turns agent traffic into long-horizon environments whose difficulty comes from
decisions that matter. A point in a trajectory counts as a decision only when competing actions
taken from that exact checkpoint, continued under one frozen policy and budget, lead to measurably
different outcomes. A family of environments qualifies only when which action is better flips
across coherent variants of the world. Everything here is a Python API; three stages, and the
whole pipeline (see [End-to-end pipeline](#end-to-end-pipeline)), are also `exp eval`
continuation actions.

## Data roles

`exp/optimize/evaluation/variance/exposure.py` fixes a role for every leakage lineage before any
tag is bought:

| Role | Lineages | Read by |
|---|---|---|
| `final_evaluation` | every held-out lineage | nothing in this pipeline |
| `world_learning` | fit lineages, seeded hash split | rules stage, world-model construction, retrieval |
| `mining_probes` | the rest of the fit lineages | actor-critic runs, decision families, milestone comparisons |

Tag selections (`exp/simulation/mining/tag_selection.py`, over the decision-model tag matrices in
`exp/simulation/mining/tagging.py`) apply only inside a role. `CorpusScope`
(`exp/simulation/retrieval/scope.py`) restricts retrieval indexes and world-model construction to
declared world-learning traces; held-out traces can never enter a scoped index or construction.

## Variance build

`run_variance_build` (`variance/build.py`, CLI action `variance-build`) turns a completed
`exp build` into this pipeline's simulator as one saved, replayable stage:

1. `plan_exposure` splits the build's lineage bindings before any tag is bought; the plan is saved
   as a `variance-exposure-plan` artifact and written to `exposure.json` for the actions below.
2. `DecisionsTraceTagger`, resolved from the catalog decision alias (default `jev`, on a typesafe
   connection), answers a local question bank (default `capability-shape-task-v1`, read from
   `EXP_QUESTION_BANK_ROOT` or `.exp/question_banks`) over the task view of world-learning and
   mining-probe traces. An optional `observations_bank` tags world-learning observations views
   as annotation. Final-evaluation traces are never rendered or sent. Each matrix is saved as a
   `variance-tag-matrix` artifact with its bank, revision, view and digest; a trace whose answers
   cannot be bought or read is recorded as a failure and left out of retrieval, never redrawn.
3. Optional `construction_rule` and `retrieval_rule` tag rules narrow the two world-learning
   corpus scopes; by default both hold every world-learning trace.
4. The scoped, tagged fit-only index reuses the build serving index's embedder identity, chunking
   and `default_top_k`, so every vector comes from the build's embedding checkpoint. A semantic
   embedder is bound to a client that refuses calls: a missing vector stops the run.
5. `persist_scoped_world_model` rebinds the build's world model to that index. Reconstruction
   over the construction scope is off by default; the Python API runs it with an injected
   admitted constructor.
6. Every mining-probe task gets task-view query tags in a `tag-retrieval-binding` (default boost
   weight 0.25), with a weight-zero control binding for cosine-only baselines.
   Executable worlds (continuation jobs, actor-critic runs, decision families and Harbor trials)
   bind one by its reference in `GroundedRuntimeDefinition.tag_retrieval`: the artifact is verified
   against its manifest and type, the bound task's retrieval records the policy as evidence, and
   the binding joins the definition identity, so a tagged checkpoint never resumes untagged.

The dry run prints traces per role, decision calls per view (distinct traces x
ceil(questions/32)) and how many world-learning embedding inputs the build already saved. Decision
calls are priced from the catalog when the alias has prices and are otherwise unpriced. The
`variance-build-report` lists exposure counts, tagging coverage and failures, the index, the world
model, both bindings and the world-learning traces for the rules stage.

## Rules stage

`run_rules_stage` (`exp/optimize/evaluation/rules/stage.py`, CLI action `rules-stage`) learns how
the environment behaves from world-learning traces only, keeping rules apart from state:

1. `parse_tool_contracts` reads tool contracts from traces without a model call.
2. `research_in_parallel` (`rules/research.py`, the default `agentic_research` profile) researches
   every trajectory with bounded agentic draws, one per assigned prompt variant
   (`contract_first`, `state_transition_first`, `failure_and_error_first`), concurrently up to
   `concurrency`. Each draw plans, reads with read-only tools (its own steps, tool contracts,
   related steps of the other admitted world-learning traces, the registered world-learning
   retrieval, earlier rounds' notes), drafts, self-checks and answers with rules, open questions
   and a research history (questions asked, resolved or not; hypotheses confirmed, refuted or
   open). Rules cite only their own trajectory; history notes may also cite what the draw read.
   Anything else is refused and kept as a failure, never redrawn. Each trace's history is saved
   as a `rule-research-history` artifact (JSON plus Markdown). The explicit `single_draw` profile
   keeps the earlier path: `extract_rules_in_parallel` makes one forced draw per trajectory.
3. `merge_rules` groups proposals, keeps contradictions and unknowns, counts distinct
   trajectories and decides no support; held-out traces are excluded. `weigh_rules_in_parallel`
   (`rules/weighing.py`) then weighs each rule in its own agentic draw: supported, contradicted
   or unknown from the strength and independence of the cited evidence, with a rationale. Code
   checks only structure (`check_weighing`: citations are the rule's own locators, none reach an
   excluded trace, a decided status cites its side). Witnesses (native or specification) attach
   by rule id.
4. The rule book is saved as Markdown (`README.md`, `tools/<tool>.md`, `environment.md`) plus
   `rules.json` in one `rule-book` artifact. Under `agentic_research`, `condense_memory`
   (`rules/condense.py`) condenses the book and every research history into the
   `world-model-memory` artifact (`memory.md` plus `memory.json`) bound for the world model at
   runtime; every contradiction, unknown rule, open question and unresolved research question is
   kept, by the host when the condenser leaves one out.
5. Properties are checked against simulated sessions and boundary workloads challenge numeric and
   membership boundaries; a contradicted rule with no independent witness keeps every family that
   relies on it diagnostic.

### The `jev_memory` profile

Registering `memory` settings (`MemorySetSettings`, with the System One `decision_alias`) makes
`jev_memory` the default profile; `agentic_research` and `single_draw` stay available by name.
`run_rules_stage` then takes the System One decisions client (`decisions`, a
`DecisionsTraceTagger` or `DecisionBoundary`) and replaces steps 2 to 4
(`rules/memory_stage.py`):

- **Research as System Two plus System One.** Per topic (each tool, plus the environment), a
  System Two draw (`rules/system_two.py`, the shared agentic loop, stances `preconditions_first`,
  `state_effects_first`, `failures_first`) sets explicit evidence objectives with
  `ask_system_one`. System One (`rules/system_one.py`) cuts the world-learning traces into one
  region per step and answers each objective with parallel typed Jev judgments per region: Noul
  `shows`, Score `relevance`, Choice `reading` (stop, read the surrounding steps, inspect more)
  and Noul plus Choice for a lead of another kind. Reading adapts round by round until enough
  passages are found or the bounds stop it, and returns the original passages with locator,
  trace and step plus discovered leads, which a second round follows. Decisions are bought once
  through the project decision cache at bounded concurrency; every objective's judgments and
  passages are saved as `jev-objective-findings`, every draw as `topic-research`, and the run
  reports a research coverage manifest.
- **A memory set of small files.** Researched rules merge as one `add` proposal each into a
  `rule-memory-set` (`exp/simulation/rules/memory_set.py`): `rules/<topic>/<rule-id>.md` per rule
  (truism, why it holds, quoted passages with locators, status with the weighing rationale,
  applicability, open questions), `index.md`, `tools/<tool>.md`, `contradictions.md`,
  `unknowns.md`, `memory-set.json` and `files.json` with every file's digest and the set digest.
  The existing weigher sets each rule's status as a `patch` proposal.
- **Inspector governance.** Single-responsibility inspectors (`rules/inspectors.py`:
  `evidence_auditor`, `contradiction_auditor`, `coverage_auditor`, `fidelity_auditor`,
  `reasoning_auditor`, `redundancy_auditor`), each with its own prompt variants, findings and
  edit permissions, run for `governance.cycles` cycles (default 3, `rules/governance.py`). They
  propose small patches to individual files against each file's digest, with a rationale and
  citations; the merge applies each proposal atomically and records it, rejecting a stale patch
  as a conflict. Coverage objectives go back to System Two; the fidelity auditor reads held-in
  replay misses (by default the call-free `PredicateReplay`). Each cycle saves a version, and a
  cycle with no proposal ends governance early.

The governed set becomes the rule book, so the checker and challenges are unchanged.
`memory_stage_costs` prices System Two draws, System One decisions, inspector draws and the
per-rule weighing. At runtime `GroundedRuntimeDefinition.rule_memory` pins the set and a
`RecallPolicy`: before each prediction `RuleRecall` (`exp/simulation/rules/recall.py`) scores
only the index rows of the called tool and the environment (Score: governs this call; Noul:
applicability met) and attaches the passing rules' truism, reasoning, status and rationale to the
called tool within a byte budget. Recall decisions replay through the decision cache, every
step's receipt is recorded in its prediction intent, and the definition digest covers the set.

A bound rule book makes the world model a state machine: the model proposes each transition and
`check_proposed_transition` (`exp/simulation/rules/checker.py`) decides. A rule-breaking proposal is
re-asked with the broken rules as feedback and never committed. Durable sessions take an opt-in
`transition_checker`; executable worlds bind the rule book into `GroundedRuntimeDefinition`.

## Screening: judges, critics and continuations

`run_actor_critic` (`variance/actor_critic.py`, CLI action `actor-critic`) runs G attempts of a
mining-probe task, picks one trajectory by seed, lets J point selectors (`judges`) choose up to K
points and C critics estimate the success probability at each, then runs M continuations per
point. Point selectors, critics and the final episode judge are separate roles even when they
share a model. Point selectors and critics read a privileged packet (every attempt, its historical
mechanical verifier grade, the group's historical verifier rate and the verifier specification,
each labelled a historical diagnostic) and both carry the eval-audit checklist. Every
continuation is judged by the spec's frozen `episode_judge` (`judge_study`), and only those
verdicts form the measured rates: critics are scored by squared error against the judged rate of
each point's scored continuations, and a critic with nothing judged pass or fail is unscored. The
selectors' spread score is a screening heuristic only and never a training reward. Every measured
rate records its `EstimateMeaning` (actor, harness, budget, sampling, runtime world digest and
judge definition identity).

Error mining is a versioned experiment (`protocol: "error-mining-v1"`).
`ActorCriticSpec.error_mining` registers the reference counts G=16, J=8, C=8, K=4, M=8; every
count, model setting, budget, seed, `references` entry and `actor_tier` stays an explicit field.
The pick is a seeded uniform choice over the eligible population only (completed attempts with
checkpoints and enough remaining calls); the report records the rule, seed digest, order and why
each other attempt was ineligible, plus planned and actual populations and measured usage per
role draw. The evidence packet addresses trace steps, actor outputs, grader evidence, checkpoint
prefixes and registered references by locator and names every reference kind that is
unavailable.

`selector_profile` and `critic_profile` are `EvidenceToolProfile`s (`variance/evidence_tools.py`):
`forced_response` (the field default: one model call, no tools), `read_only_tools` (read, then
answer) or `agentic`, which `error_mining` registers by default (`AGENTIC_PROFILE`). An agentic
draw plans (`record_plan`), investigates with the read-only tools, submits a draft
(`submit_draft`), sees the host's validation of that draft in a self check that may read more
(`record_self_check`), and then gives its forced final answer. Every draw runs under an
`EvidenceBudget` (model calls, tool calls, input and output tokens, wall clock, dollars); a phase
runs only when there is room for it and every later mandatory phase, otherwise it is skipped and
recorded, and a draw that cannot afford its final answer stops as `budget_exhausted`.

The tools read only frozen documents: trace steps, actor outputs, grader evidence, checkpoint
prefixes, the saved world state at each offered point (by JSON pointer), the public task and
harness definition, declared tool schemas with their rule-book contracts, registered references,
side-by-side attempt comparisons, divergence points across the group and, when a
`traceml-labels` artifact is registered as a `traceml_labels` reference, step labels and moves.
`retrieval` registers a `TraceSearchSpec`: tag-aware retrieval (with the `TagRetrievalBinding`)
over a fit-only index restricted to a world-learning `CorpusScope`; held-out partitions, another
scope, or any scoped trace outside world learning are refused before any search. Unavailable
material is refused by name. Nothing is executed and nothing reaches the actor.

Each draw is assigned one prompt variant (`variance/role_prompts.py`): selector stances
`divergence_seeking`, `failure_forensics`, `counterfactual_branching` and `pivot_and_return`;
critic stances `base_rate_anchored`, `mechanism_tracing` and `adversarial_skeptical`.
`selector_prompts` and `critic_prompts` weight them; draws are apportioned by largest remainder
and shuffled by the spec seed, and the transcript records the variant with its stance and phase
digests. `run_analysis` saves every request (with its phase), tool result, plan, draft, self check
and final answer as an `AnalysisTranscript` that replays without calls. The CLI cost inventory is
an upper bound: each selector and critic row is priced at `budget.model_calls` requests at the
role's attempt ceiling (12 for `AGENTIC_PROFILE`), the group at its own ceiling, `J * K` suffix
jobs of M continuations at the continuation ceiling, and the episode judge at
`judge_model_calls(profile)` requests per episode over `G + J * K * M` episodes.

### Agentic roles and prompt sets

Every model role on this path that investigates evidence runs one versioned prompt set per draw
and saves every phase; the set's digest is part of the draw's identity, so a prompt change is a
new draw, never a silent replay.

| Role | Owner | Prompt sets | Phases |
|---|---|---|---|
| Point selector, critic | `variance/role_prompts.py`, `variance/evidence_tools.py` | Stances assigned by `assign_prompts` | plan, investigate, draft, self check, final |
| Episode judge | `variance/judge_prompt.py` | `episode-judge-prompt-v1` or an agentic stance (`DEFAULT_AGENTIC_JUDGE`) | plan, investigate (criterion checks), self check, final |
| Rubric author | `variance/rubric_agent.py` | `requirements-first`, `failure-patterns-first`, `alternatives-first` | plan, draft, critique, final, then reconciliation |
| World author | `variance/world_agent.py` | `contract-first`, `state-machine-first`, `failure-mode-first` | plan, draft, bounded repair, final |

### Selector training objective

Point selectors are trained on how decision-relevant their picks proved, never on their
screening spread score. `SelectorObjective` (`variance/rewards.py`, `selector-decision-variance-v1`)
scores one draw with `selector_score`: the mean judged outcome variance `4p(1 - p)` over its picks
with at least `minimum_scored` (default 4) judged pass or fail continuations, minus
`duplicate_penalty` per repeated pick and `unusable_penalty` per pick with no continuations, each
as a share of the draw's picks. `selector_records` (`variance/records.py`) builds one
`SelectorRecord` per selector draw of a saved run from its judged continuations only, as one
group-relative group keyed by the run's group and pick. The group is refused, with each reason
recorded, when the run's episode judge is not the expected selector judge, a point was judged by
another judge or for another study, judge coverage fails the admission policy, or fewer than two
draws were rewarded; a record is trainable exactly when nothing refuses it.

### Training rollout schedule

`variance/schedule.py` (CLI action `variance-schedule`: status, a plan that may be executed, or
a revision registration) decides which error-mining rollout runs next and never trains. During warm-up
every rollout runs the base registration's J selector and C critic draws. `TransitionRule` moves
to steady state after a number of warm-up rollouts, after both the critic Brier score and the
selector reward plateau (`PlateauRule`, measured on the current epoch's saved records), or either.
In steady state every `steady_every`-th rollout runs `steady_draws` (15) selector and critic draws
and the rollouts between are actor-only, still judged by the frozen episode judge.
`register_revision` records a new frozen snapshot of a role and starts a new epoch; records never
mix epochs. Every plan, completion and revision is an immutable event chained to the one before,
so the schedule resumes and replays from disk without a model call.

## Decision families

`run_decision_family` (`variance/family.py`) takes one screened point through:

1. Decision-state and scenario authors (`variance/authors.py`): options executable as competing
   actions, variation axes with permitted values grounded in rules or observations, invariants,
   single-proposition criteria. Scenario sets must pass `check_scenarios` (option identities,
   permitted values, invariants compared before and after, distinct (state, obligation)
   identities, the instruction axis, briefing leaks, an observation witness for every deciding
   fact within the remaining calls).
2. Coherent variants (`variance/variants.py`): fork before the first turn that revealed an edited
   fact, edit grounded world state there, check public context, instance facts and clock or pending
   events, freeze every restorable state part and prove restoration.
3. Executed witnesses and option contrasts (`variance/contrast.py`): the witness calls must show
   every deciding value without stating the answer; each option is forced from the same checkpoint
   through the preregistered action-contrast experiments, and every cell is judged once by the
   variant's frozen judge. When the variant's `VariantJudging` binds a world freeze, the frozen
   variant is what runs: `frozen_start` (`variance/admission.py`) returns the freeze's saved start
   checkpoint and the runtime definition whose digest the freeze recorded, the witness and every
   contrast start from that checkpoint under the family job rebound to that runtime, and the
   edited decision checkpoint of step 2 is coherence evidence only.
4. `family_flip`: at least two variants with separated, different winners that match their declared
   correct options; a contradicting winner goes to triage and blocks qualification.
5. Fresh confirmation on independently seeded samples under the same frozen judge, then the
   judge's coverage of the confirmation episodes (passed, failed, unassessable, malformed,
   unavailable, without evidence, insufficient evidence, rubric problems). Probabilistic criterion
   grading (`variance/grader.py`) may still run, as calibration evidence only; an uncertain
   criterion never decides eligibility.

**Initialization modes.** A scenario starts by `resume` (the coherent actor-visible prefix and its
remaining calls) or `fresh_briefing` (an authored `FreshBriefing` at the fork, with the same
remaining calls, restored under the original environment task). A briefing is leak-checked
against an `ActorView` of what the actor had seen: it may not state the intended option's action,
old or new values at edited paths, or decision-relevant values the actor never observed. Without
a view a briefed scenario is refused. In a family, `DecisionFamilySpec.fresh_briefing` passes the
point's `ActorView` to the scenario author and its checks, and `required_state` names state parts
(for example `files`) that every variant checkpoint must hold beyond the baseline; both leave the
draws identity unchanged while unset.

**Instruction axis.** Instruction variation is its own declared axis
(`DecisionState.instruction_axis`), separate from state axes and enabled by
`instruction_variation`. Each value is `original`, `reworded`, `goal_change` or `consent_change`. A rewording is `equivalent_to` another value and changes no obligation,
so a variant differing only by rewording is refused as decision variance; goal and consent
changes cite their authorization and bind their own criteria. Any non-original instruction must
start by `fresh_briefing`, since a resumed prefix already stated the original.

`TrainingEligibility` (`variance/records.py`) is computed, never asserted, and lists every failed
condition; training records exist only for eligible families. The family's
`FamilyAdmissionPolicy` (`variance/admission.py`) adds, per confirmed variant, judge coverage,
rubric problems, rubric compatibility, world freeze and world provenance. World provenance needs
more than matching digests: every selection and confirmation contrast must have started from the
freeze report's saved checkpoint for the variant, under the runtime definition the freeze
recorded, with a judge bound to that checkpoint's task and that runtime; an unfrozen variant
records a failed `world_provenance` condition. `export_family_harbor` packages a
confirmed variant for Harbor, diagnostic unless the family is eligible and an executable
qualification is supplied.

## Authoring and freezing worlds

`build_authoring_packet` (`variance/authoring_packet.py`) binds, for one decision family, the
original trace and evidence sources (checked against the exposure plan, so no final-evaluation
lineage reaches an author), the saved rule book with each rule's status, contradictions and open
questions, the exact task digest (`variant_task_sha256` over task, instruction and state) of every
variant, the pre-decision checkpoint and the decision brief, whose claims stay hypotheses. It is
the only input of both authors.

- `author_world` (`variance/world_author.py`) returns a `WorldSpec`: grounded runtime fields, rule
  bindings, tool contracts, per-variant initializations, evidence hooks and induced transition
  programs in the `transition(state, args, facts)` ABI, which induction validation must accept.
- `author_rubric` (`variance/rubric_author.py`) returns a `Rubric` of atomic criteria with
  applicability, source support, evidence needs, anchors, valid alternatives and a role
  (`required_correctness` or `efficiency_diagnostic`); calibration cases are build-time controls,
  never shown to the judge. `save_rubric` stores it as an `episode-rubric`, and
  `judge_definition` binds it to a `JudgeProfile` configured before authoring.

Each author is one forced draw on the shared author-response owner, bought once and replayed. It
never reads the other author's output, and it answers either ready or `needs_research` with
explicit research requests and no invented artifacts.

`freeze_world` (`variance/world_freeze.py`) checks the simulator, never an actor. It runs the
draft's fixtures and the rules stage's boundary workloads, probes every load-bearing precondition
of a state-writing tool (a silent write under missing or failing authorization rejects the world),
and records a `VariantFreeze` per variant: `frozen` (may qualify), `diagnostic` (an unsupported
boundary) or `rejected`, with the start checkpoint proven to restore exactly and the digest of the
variant's grounded runtime definition (`definition_sha256`). `frozen_run_digests`
(`variance/admission.py`) returns the task and world digests a run from that frozen start
produces, which are what the variant's judge definition binds.
`review_compatibility` records every disagreement between the world and the rubric without
reconciling them.

### Coalition evidence, agentic rubrics and revision

The packet carries the behavioral coalition (`variance/coalition.py`, `mining_coalition` and
`contrast_outcomes`): every grouped attempt of every tier with its judged label (a label from an
earlier discovery judge is `historical`), distinct mistakes, recoveries and valid alternatives,
selector choices and rationales, critic estimates beside the judged rates they predicted, judged
contrast outcomes per option and optional TraceML step labels, each with its locator and evidence
status. Tiers are never pooled, attempts or labels from final evaluation are refused, and no
criterion may name a tier or actor model. `author_rubric_agentic` (`variance/rubric_agent.py`)
runs the rubric prompt variants independently, critiques each draft against the host's design
checks (authoritative and fed back), and reconciles at least two ready drafts; a merge that drops
any disagreement `host_disagreements` found is refused as a silent resolution.

### Behavior library, labeling, curation and the design critic

Rubrics start from a versioned behavior library (`variance/behavior_profiles.py`) instead of
rediscovering the same behaviors per family. A version holds bad-behavior linters (detection
guidance, the evidence a finding needs, labeled positive and negative examples with locators,
false-positive notes), cross-task behavior profiles and task-scoped criteria; each version is its
own `behavior-library` artifact citing its parent and the evidence behind every change. In the
pipeline the library is on by default (`authoring.library_id`, `agent-behavior`). For every
family, before its rubric is authored:

1. **Labeling** (`variance/behavior_labeling.py`, `label_behaviors`): the labeler prompt variants
   (`authoring.labeler_variants`) read the family's development packet, never final-evaluation
   rows, and propose behaviors with exact locators; a refused answer gets one repair draw.
2. **Curation** (`variance/behavior_curation.py`, `curate_library`): when labels exist, the
   curator (`authoring.curator_variant`) plans, then proposes typed operations (merge, add
   evidence, split, narrow, reject) that the host applies by structure into the next version.
   The family binds the version its own curation produced, so a resumed family never picks up a
   version a later family saved.
3. **Authoring and revision**: `author_rubric_agentic` and `revise_until_stable` receive that
   version. The rubric records one decision per offered entry, cites the version, and copies the
   judge guidance and examples into `behavior_guidance`, which the judge reads and actors never
   see. Before authoring, every actor-visible text of the family (the task, instruction values
   and briefings) passes `refuse_actor_exposure`, so no judge-only guidance reaches an actor.

The design critic (`variance/rubric_critic.py`, `DesignCriticProfile`) is on by default
(`authoring.rubric_profile.critic`): several critic variants review every drafted and revised
rubric against the design checks with forced, cited verdicts, and the host refuses every
violation. The plan prices labeling (two draws per labeler variant), curation (three draws),
authoring including critic reviews (`RubricAuthoringProfile.draws`) and one reviser plus critic
review per revision round. Judged criteria have three roles: `required_correctness` decides
success; `efficiency_diagnostic` and `behavior_diagnostic` are reported and never decide it.

`variance/rubric_revision.py` revises a rubric from judging. `judged_problems` collects
judge-reported rubric problems, calibration misses, coverage gaps attributable to the rubric and
compatibility disagreements (an unjudged target blocks acceptance but is never sent as a rubric
fault). `revise_until_stable` runs at most `max_rounds` forced revision draws; each revised rubric
passes the author's host checks, is saved as a new `episode-rubric` and is bound to a new
`JudgeDefinition`, and the calibration episodes and affected evidence are re-judged under it. A
revision is accepted only when no problem remains; otherwise the family stays diagnostic under its
original judge. `require_current_labels` refuses labels made under an earlier definition.

### Served programs and the agentic world author

A `GroundedRuntimeDefinition` (`executable/grounded.py`) may bind validated transition programs
(`programs`, an `InducedCoverage`). Covered tools then run as exact sandboxed code inside their
validated domain, and every other call, including one outside a program's domain, is predicted by
the grounded model under the rule book. The programs are part of the definition, so its probe
digest, the judge's world digest and the freeze's `definition_sha256` all cover them.
`world_runtime_definition` binds a ready world's programs, and the world author refuses programs
the served runtime would never run. Simulated state and history stay in the predictor image part;
the programs' partition is a separate part that variant edits (`variance/variants.py`) keep.
`author_world_agentic` (`variance/world_agent.py`) runs the world prompt sets, each planning,
drafting and repairing from the exact host-check and sandbox-validation failures within bounded
rounds; every round is bought once and replayed, and each final draft is an ordinary
`world-draft` that `freeze_world` freezes unchanged.

### Executed rule challenges

`run_challenges` (`rules/challenges.py`) executes boundary workloads through a
`ChallengeExecutor`: `ProgramChallenges` runs a tool's validated program in the sandbox (other
tools stay unexecuted unless an observed transition is supplied) and `RuntimeChallenges` runs the
bound served runtime. Each case is triaged against its rule. A rule is `supported` when an executed
case agreed and none contradicted it; a contradicted rule without an independent witness becomes
`contradicted` and listed in `diagnostic`, so families on it stay diagnostic; a contradicted
independently witnessed rule keeps its status and is a `world_defects` entry. Executed witnesses
are recorded as simulated `challenge:` evidence, never native. Results are saved per case and
replay without execution. The rules stage runs this behind an explicit `challenge` executor, and
`freeze_world` runs it for every variant's initial state, recording `challenge_rules` and the saved
`challenges` report; a world defect rejects the world.

## Judge and evidence contracts

`JudgeProfile` (`variance/episodes.py`) fixes the judge model alias and snapshot, prompt version
and digest, output-schema digest, temperature, output allowance, reasoning effort,
`evidence_tool_budget` and `maximum_input_bytes` (default 128000, the shared author default).
`JudgeDefinition` adds the rubric artifact, `rubric_sha256` (the digest of the rubric file's
content, refused when it equals the artifact's manifest digest), the event contract
(`episode-events-v1`), the task digest and the world digest. Any change is a new definition and
new labels.

Both digests describe a run, not an authoring artifact. `run_task_sha256`
(`variance/episode_evidence.py`) digests the starting checkpoint's actor task and environment
task; `run_world_sha256` digests the runtime configuration the episode actually ran under, which
for a frozen variant is its grounded runtime definition (the freeze's `definition_sha256`; the
authored world's own digest stays the freeze's `world_sha256`, which the compatibility review
covers). Every evidence converter derives `task_sha256` and `world_sha256` this way from the
checkpoint and the job configuration it was given, never from the judge definition.
`require_run` (`variance/judged.py`) refuses a study or contrast whose digests differ before any
judge call, and `judge_episode` refuses any evidence that does not match its definition. The
authoring packet's `variant_task_sha256` identifies a packet variant and is not a judge binding.

`EpisodeEvidence` is ordered and append-only. Each `EpisodeEvent` names its kind, its owner
(`actor`, `user`, `simulator`, `world`, `harness` or `experiment`) and its scope: `context` events
(the inherited prefix) come before every `scored` event. `variance/episode_evidence.py` builds it
from a continuation branch, an action-contrast cell (the imposed action is scored and owned by
`experiment`) or a Harbor trial ledger, listing missing replies, unanswered calls and interrupted
branches.

## Milestone comparison

`run_selection_comparison` (`variance/milestone.py`) preregisters judge-chosen points against
uniform, late-failure and midpoint baselines at an equal point budget and counts how many turn out
to be decisions. `compare` reports `judge_better` only when the paired lower bound beats every
baseline; otherwise `not_shown`.

## Reward ownership

One frozen LLM episode judge is the only semantic success authority on this path.
`judge_episode` (`variance/judge.py`) judges one complete `EpisodeEvidence`
(`variance/episodes.py`) with one `JudgeDefinition` (a `JudgeProfile` bound to the authored rubric,
event contract, task and world) and saves an `episode-assessment` artifact whose inputs are the
saved `episode-evidence`, the rubric and every judge response. `episode_reward` maps its verdict to
pass 1, fail 0, anything else unscored. Downstream records cite the assessment (`JudgedEpisode`);
they never copy a success flag.

The judge renders the whole evidence when it fits the frozen input budget. Otherwise a profile
with an `evidence_tool_budget` receives an inventory of every event (ids, kinds, owners, scopes,
sources, content sizes) and reads contiguous event ranges in full through `read_events` before
its one `assess_episode` answer; it may cite only events it read and may pass only after reading
every scored event. Without that budget the episode is saved unscored as insufficient evidence.
Evidence is never truncated, graded in chunks or averaged. Code validates shape, rubric coverage,
the verdict rule over the judge's own findings and every citation; a malformed or unobtainable
answer is saved `malformed` or `unavailable`, unscored and never redrawn. Assessments replay
exactly by protocol, definition identity, evidence identity and input budget. The prompt sets and
schemas live in `variance/judge_prompt.py`; `judge_profile(prompt_set=...)` binds one set, so the
choice is part of the definition identity. `episode-judge-prompt-v1` is the single-pass prompt.
The agentic sets (`DEFAULT_AGENTIC_JUDGE` and two other stances) run one judgment in phases: a plan
of the evidence each criterion needs, range reads, working per-criterion checks, a self check
against host validation (missing criteria, unread citations, unread scored events, verdict-rule
contradictions) that may read more ranges within the same budget, and the one final
`assess_episode`. Every phase is saved in the assessment's `inspection.json`; one judgment makes
at most `judge_model_calls(profile)` draws (`evidence_tool_budget + 4` for an agentic set).

Every place on the execution-to-reward path that derives a success bit, an empirical rate, a
critic target, an advantage, an eligibility result or a Harbor reward, in its migrated state.
Paths are relative to `exp/optimize/evaluation/`; function names are authoritative.

Semantic labels on this path read a saved `EpisodeAssessment`:

| Site | Reads | Notes |
|---|---|---|
| `variance/judged.py` `judge_study`, `judge_cells` | `judge_episode` over evidence from each continuation branch or contrast cell | One `JudgedSlot` per planned episode: an assessment reference or the reason no evidence could be built. `require_run` checks the run's own digests first |
| `variance/judged.py` `slot_rewards`, `slot_coverage`, `judged_metrics` | `episode_reward(load_assessment(...))` per slot | Pass 1, fail 0, else None; unscored episodes stay in the planned denominator and in `JudgeCoverage`, never 0.5 |
| `variance/actor_critic.py` `_continue` (`PointOutcome`, `group_advantages`) | `slot_rewards` of the point's judged study | Critic Brier scores, selector spread scores and ranking are unchanged arithmetic over these judged rates |
| `variance/contrast.py` `run_option_contrast`, `option_outcome` | Judged cells of every action contrast | Each cell is judged once as the imposed action (`owner=experiment`, `scope=scored`) plus its suffix; Wilson and separation bounds come from judged labels |
| `variance/contrast.py` `variant_verdict`, `family_flip` | Option outcomes above | A mismatch with the declared correct option stays a triage finding, never a reward |
| `variance/family_stages.py` `grade_variants` | `OptionContrast.coverage` of each confirmation | Judge coverage per confirmed variant; criterion grades from `variance/grader.py` are calibration evidence only |
| `variance/records.py` `family_flip`, `fresh_confirmation`, `critic_scores` conditions | Judged contrasts and judged rates | Computed over the frozen variant judge's labels |
| `variance/records.py` `judge_coverage`, `rubric_problems`, `rubric_compatibility`, `world_freeze`, `world_provenance` conditions | `variance/admission.py` `judging_problems` | Replaces the old `rubric_unambiguous` condition; one uncertain criterion is not proof of ambiguity |
| `variance/records.py` `training_records` | Judged confirmation cells of each confirmed variant | Groups cite every assessment and read advantages back from them; discovery critic records stay diagnostic and untrainable |
| `variance/exposure.py` `estimate_meaning` | `JudgeDefinition.identity` | Every estimate binds the judge, so a new rubric or profile is a new estimate |
| `executable/trial.py` `result` (`_judged_result`) for a task with a `judge` | `judge_episode` over `ledger_evidence` of the closed trial ledger (`executable/ledger.py`) | Reward is `episode_reward`, emitted only when the observable contract is qualified; `family_executable_task` always binds the variant's judge |
| `harbor/client.py` `reward.json` | The backend's judged `reward` and `status` | Encoding only |

Mechanical code that stays as simulator and protocol validity, never actor reward:

| Site | What it decides |
|---|---|
| `continuations/study.py` `_run_branch` (`VerifierGrade` on `ContinuationBranch.assessment`), `known_outcomes`, `ContinuationStudy.metrics` | Recorded as simulator and protocol diagnostics of the general continuations owner; nothing on this path reads them for a label |
| `continuations/action_contrast.py` `_assessment`, `_metrics` | Mechanical arm diagnostics of the general contrast owner; decision-variance labels come from `judge_cells` |
| `variance/evidence.py` `group_rate`, attempt grades, `success_rate` | Historical verifier diagnostics shown to point selectors and critics, labelled as such in the packet |
| `variance/evidence.py` `candidate_branches`, `variance/actor_critic.py` `_pick` | The eligible population (completed attempts with checkpoints and enough remaining calls) and the seeded pick over it |
| `variance/contrast.py` witness `_verdict` | That every deciding fact is discoverable within the remaining calls, without stating the answer |
| `variance/records.py` dispositions, tool coverage, rule witnesses, policy consistency | Audit and world evidence; policy consistency excludes the runtime world, which `world_provenance` checks per variant |
| `variance/world_freeze.py` `freeze_world`, `review_compatibility` | Whether the authored world is a valid simulator and how it disagrees with the rubric |
| Branch `status`, `actor_stop`, cell `failure` | Protocol facts; an outcome-unknown branch is unscored, an actor budget stop is still judged |

Legacy modes that keep their mechanical contract and are never used for decision-variance labels:

| Site | Mode |
|---|---|
| `executable/trial.py` `result` for a task without a `judge` | Observable mechanical grade of an executable task outside this path |
| `harbor/trial.py` `result` | Mechanical grade combined with dual-evaluator axes for `EvaluationTask` Harbor trials |
| `variance/grader.py` `grade_episodes` | Probabilistic criterion grading kept as calibration evidence |

## TraceML step labels

`exp/simulation/mining/traceml.py` labels every step of a trajectory (a tool call with its result,
or a visible message) against the `ActionLabel` taxonomy through the existing decision tagging
seam, over the `step` view: one unit per step (`step_units`) with the task, a few earlier steps
and the focal step, row identity `<trace_id>::<locator>`. Pivots (a strategy change prompted by
evidence) and returns (back to an earlier state or an approach left before a strategy change) are
judged by a model (`exp/simulation/mining/traceml_moves.py`): each window of labelled steps, with
its calls, results, labels and mechanical state evidence, is judged under several reading stances
with forced, cited answers; code checks only coverage and citations, and a move is kept by strict
majority of the usable judgments (minority claims are recorded as contested). The pipeline's
`traceml.move_judge` is required whenever the stage runs, and every draw is saved and replayed.
With `traceml.error_bank`, each tier's attempts are also error tagged as wholes (the coalition's
mistake patterns) and step by step (error-ranked points, recorded for review). Uncertain steps
stay in every count and refused traces are listed in the coverage manifest. The saved
`traceml-labels` artifact (`persist_traceml_labels`) carries labels, moves, coverage and per-tier
comparisons; selectors, critics and rubric authors may read it, and `candidate_points` maps moves
to actor-critic checkpoint indices.

## Fresh briefings

A fresh briefing's entitled paths are checked in code (they exist, are not edited and show values
the actor observed). Whether the briefing text leaks the intended option, an edited value or
unobserved knowledge is judged by the model leak reviewer (`variance/briefing_review.py`) under
three reading stances, with findings that quote the briefing and cite the actor's view; any
finding or unusable review refuses the scenario set.

## Cross-talk probe

`run_crosstalk_probe` (`variance/crosstalk.py`) measures whether in-context exemplars from one
environment move judged scores in another. The source job runs N attempts labelled by the source's
frozen judge; the `top_k` best become treatment exemplars and a seeded draw of completed attempts
is the random control, both rendered with identical framing. Each target runs `none`, `random` and
`top` arms that differ only in the delivered exemplars (`context` through a fresh initial
checkpoint, so the target's task and world digests are unchanged, or a caller-supplied `file`
delivery). Each target's own frozen judge labels every arm; branch `i` pairs across arms, unscored
episodes stay in coverage and leave their pair unpaired. `crosstalk_verdict` reports cross-talk
only when the paired-bootstrap lower 95 percent bound of both `top - none` and `top - random` is
above zero. The spec is preregistered before any call and a rerun replays the saved result.

## End-to-end pipeline

`run_decision_variance` (`variance/pipeline.py`, CLI action `decision-variance`) runs the whole
sequence from one `DecisionVarianceSpec` (`variance/pipeline_spec.py`) in six stages, each run
through the owners described above:

| Stage | Spec slice | Owner and saved outputs |
|---|---|---|
| `variance-build` | `build` | `run_variance_build`: exposure plan, scoped world model, scoped index, tag binding |
| `rules` | `rules` | `run_rules_stage` over world-learning traces, boundary workloads executed in the bound world; the rule book is bound into the mining runtime before any probe |
| `traceml` | `traceml` | Each tier's group of attempts labelled step by step |
| `error-mining` | `mining` | The mining rubric, then one `run_actor_critic` per actor tier and its judged group |
| `families` | `authoring` | Coalition, packet, independent agentic authors, freeze, judges, families, revision, admission |
| `records` | `records` | Training and selector records, qualification, Harbor exports, cross-talk |

Each stage saves one `decision-variance-stage` record. Its identity binds its spec slice, the
identity of every earlier stage, the code revision and anything outside the spec it reads (a
TraceML bank's content; the rubric author's settings when the mining rubric is authored), so a
changed field re-keys that stage and every later one while the earlier ones replay. A rerun
replays saved stages with no call, and an interrupted stage resumes through its owners' saved
draws. `plan_decision_variance` prices every unsaved stage as a credential-free upper bound; a run
with unsaved stages needs one `PipelineApproval` of that exact plan, and a stage whose bound
exceeds what is left of the approved ceiling never starts. Measured provider attempts (and spend,
when an operation ceiling is installed) are recorded per stage. The open-weight actor council
(`COUNCIL_ACTORS`) is the default tier set, each actor its own tier and never pooled.

What the stages read beyond their own owners lives in `variance/pipeline_inputs.py`:

- **World-learning search.** With `mining.retrieval` (default on), each tier's selectors and
  critics get a `TraceSearchSpec` over the build's scoped, fit-only world-learning index, its
  `CorpusScope` and, when the build saved one, its tag retrieval binding for the mining task.
  `TraceSearch` refuses an index that is not fit-only or a scope that reaches any trace outside
  world learning, so held-out traces never enter. A semantic index binds the project's catalog
  embedder role behind the admitted transport (its identity must equal the index's); a local
  hashing index needs no provider. Query embeddings are listed as unpriced in the plan.
- **Observed replay episodes.** Every authoring packet carries, as `observed_episodes`, the
  world-learning traces that call the task's tools (up to `authoring.maximum_episodes`), so the
  world author's programs are validated against real episodes. A trace's starting state is not
  recorded, so steps that read an unknown domain stay unverifiable; recorded states are compared
  exactly.
- **Authored mining rubric.** When `mining.rubric` is omitted, the error-mining stage fixes the
  judge profile, then runs the agentic rubric author (`authoring.rubric_author` with
  `authoring.rubric_profile`) over a whole-task packet: the rule-bound start (which must be the
  task's initial context), the world-learning sources, the rule book and the observed episodes,
  with a brief that proposes no option or axis and a single baseline variant. A registered rubric
  overrides it.
- **Error tags, clusters and probes** (`variance/pipeline_errors.py`). The build stage binds an
  admitted naming client for `build.naming_alias` when the build names its tag clusters, prices
  the plan's naming calls, and lists the error-tag view plan, the cluster admission and, once
  saved, the admissions and selected probes in the plan. Error mining refuses a job whose task
  the build did not select as a probe. With `traceml.error_bank`, the coalition's mistake
  patterns come from each tier's attempt error tags, one pattern per tier and proposition.
- **Rules memory and research search.** The pipeline's rules stage learns the governed rule
  memory by default (`rules.jev_memory`, the `jev_memory` profile with System One decisions on
  `rules.jev_alias`); its plan prices that profile over one topic per task tool plus the
  environment. The saved memory set is bound with its recall policy (`RuleMemoryBinding`) beside
  the rule book into the rule-bound mining runtime, from which every tier group, frozen variant
  world and Harbor export derives. With `mining.retrieval`, research draws query the same
  world-learning search the selectors use; the research profile is priced as research, weighing
  (per merged rule, unpriced until research ends) and condensation draws.
- **Decision variance.** On by default (`authoring.decision_variance`): each family measures
  every confirmed variant's choice shift, consequence and clarity from judged free continuations
  (`decision_free_repeats`), with an open-weight classifier (`decision_classifier`, default the
  scenario author) and clarity reviewer (`decision_reviewer`, default the rubric author), and
  admission requires the joint verdict.
- **Version 2 selector records.** The records stage scores each tier's selector picks by the
  joint decision-variance score of the family built from that point (`family_decisions`); a
  point whose family was measured but not admitted scores zero, and a pick with no measured
  family stays unscored. A family's eligibility names the `decision_variance` condition when the
  joint measure blocks it.
- **Nudges, the nudge ladder and variant search** (`variance/nudge_ladder.py`,
  `variance/variant_search.py`). Their owners measure a set of scenarios, or a ladder of
  instruction nudges from strongest to none, under one frozen judge and rubric with the same
  joint measure. The pipeline does not yet run them: the family owner offers no hook for a
  searched scenario set or a ladder in place of its single scenario draw, so each family still
  authors one scenario set.
- **Option actions.** The decision-state author gives each option its executable first call
  (`DecisionOption.call`). When `authoring.actions` is empty the family forces those calls
  (`option_calls`); an option without a call becomes the family's recorded failure. Explicit
  `authoring.actions` override them.

**Qualification before export.** With `records.qualification` (an independent
`evaluation-specification` and declared controls), each confirmed variant of an admitted family
is turned into its exact executable task (`family_executable_task`: frozen start, frozen runtime
and the variant's own frozen judge), executed once for feasibility, witnessed with
`witness_controls` and qualified with `qualify_executable_task`. A variant that qualifies is
exported qualified; any other is exported as diagnostic with its refusals or failure in the
stage's `HarborExport` row. Without qualification every export is diagnostic.

The CLI action reads `{"spec": "decision-variance.json"}`. By default it prints the stage plan:
each stage's identity, whether it is saved, what it will do and its upper-bound cost inventory.
With `"execute": true` it asks spend consent once for the whole plan, runs or resumes the
pipeline, and prints each stage's replay status, measured usage and summary; the full report is
saved beside the run.

## What this does not establish

Structural gates are necessary, not sufficient. Simulator continuations reduce sampling noise but
not simulator bias, so boundaries without independent witnesses stay diagnostic. Executable
qualification does not cover native fidelity, rubric judging, tier separation or containment.

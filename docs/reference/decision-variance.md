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

### Environment families

A tag cluster is a grouping hypothesis. With `environment_families` registered
(`EnvironmentFamilySpec`, needing clustering and an injected reviewer passed as
`run_variance_build(..., reviewer=...)`), every cluster of world-learning traces is reviewed by an
agentic compatibility reviewer (`exp/simulation/mining/environment_review.py`). Each prompt variant
(`contract_first`, `state_first`, `semantics_first`) starts from the members and the tool contracts
parsed from their traces with no model call, reads with read-only tools (`read_trace`, `read_tool`,
`compare_traces`) within `max_turns`, and submits a verdict citing locators it was shown; a
reconcile draw writes the group's verdict and must give a dissent entry for every review it
overrides. Code checks only structure (citations, a split's partition, an uncertain verdict's open
questions), and a failed draft is repaired within `repairs`.

- `compatible`: an `EnvironmentFamily` with its members, reconciled contracts, state domains,
  world semantics, tasks and starting states, evidence and status.
- `split`: each proposed sub-group is re-reviewed, recursively up to `max_depth`; deeper
  sub-groups stay uncertain.
- `uncertain`: an `UncertainGroup` with its open questions, reported in the build report and left
  out of training unless `admit_uncertain` names it (status `admitted_uncertain`).

`scope_families` (`variance/family_scopes.py`) then builds, per admitted family and keyed by its
identity, a scoped tagged fit-only index over its members, the world model rebound to it
(reconstructed from the family's members alone when the build constructs its world, so no other
group's knowledge survives), the world-learning traces its one rule memory is learned from, its
tasks (mining-probe tasks placed by the fitted mixture's posterior over their source trace,
`assign_clusters`, only into a family whose contracts cover the task's tools) and a tag binding
over those tasks, all recorded as `VarianceBuildReport.environment_families`. Placed tasks are
saved in the scoping's `environment-families` record. The reviewer's model identity is part of the
build identity, so a changed reviewer re-reviews and a family-enabled build never replays without
one; the plan's `review_calls` is an upper bound.

`FamilySimulator` (`variance/family_simulators.py`) is the one simulator definition per family:
`learn_family_rules` learns its rules once per family and learner and replays them, `bind_runtime`
freezes the runtime template over the family's world model and rules (dropping the template task's
instance facts), and `definition_for` derives each task's definition from its own task, initial
state and instance facts. `plan_attempts` and
`run_isolated_attempts` make five actors times sixteen attempts eighty independent runs: each
restores the same saved start into a fresh, isolated world through `restore_continuation_world`,
and the start is verified unchanged.

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

## Start bootstrap from traces

`run_bootstrap` (`variance/bootstrap.py`) accepts selected tasks in compatible environment
families, without a pre-existing mining job. `family_from_simulator` preserves the shared
simulator's scoped corpus, tool contracts, state domains, frozen rule book, rule memory and
rules report. Frozen family knowledge is reused. A family without rules learns them once from
its own world-learning traces, never from a global corpus.

The bounded agent loop authors a start, reviews it for coherence and task support, and researches
missing deciding facts against the family's evidence. Unresolved or refused starts remain
explicit refusals. Every fact carries observed, inferred or unknown provenance and source
citations. Capture binds the accepted instruction, state, manifest and evaluator projection to
the frozen runtime; another instance's facts are discarded.

| Provenance | Validation |
|---|---|
| `observed` | The recorded state snapshot restores exactly. |
| `reconstructed` | The source trace prefix replays with equal observations and error status. |
| `authored` | Independent review and first-reveal checks support the generated facts. |

Only observed starts claim original-state restoration. Each accepted start becomes a preflighted `start-bootstrap-registration` with its exact checkpoint and continuation job.
A batch manifest records every registered, refused or failed task. Saved outcomes replay
without calls; failed tasks can resume without repeating completed tasks.

`freeze_batch` saves the manifest. `MiningSettings.bootstrap = BootstrapBinding(batch, task_id)`
selects its exact registered job. Both pipeline planning and execution resolve this binding.
The RULES stage verifies and reuses the family's frozen report, book and memory with no global
extraction or challenge. Bootstrap preparation has its own governed cost plan before those
registrations exist; the subsequent six-stage plan uses the resulting exact jobs.

## Paired behavioral differences

Register `mining.behaviors` with a `BehaviorDifferenceSpec`
(`variance/behavior_differences.py`) to mine directional contrasts before point selection:

```python
from exp.optimize.evaluation.variance.behavior_differences import BehaviorDifferenceSpec

behavior_settings = BehaviorDifferenceSpec(
    bank_id="completion-behavior-v1",
    author=alignment_author,
    reviewer=independent_reviewer,
    positive_tags=("changed_implementation", "verified_result"),
    negative_tags=(
        "unsupported_success_claim",
        "modified_evaluation",
        "skipped_required_step",
        "exploited_metric",
        "hid_failure",
    ),
    maximum_pairs=16,
)
spec = spec.model_copy(
    update={"mining": spec.mining.model_copy(update={"behaviors": behavior_settings})}
)
```

The local `completion-behavior-v1.toml` bank, resolved from `traceml.bank_root` or the
usual local question-bank directory, must contain exactly the declared tag identities, with
their full propositions. For example, `modified_evaluation` can mean that the focal action
changes the tests or metric used to evaluate the task. That observation alone does not make the
edit wrong. `verified_result` should require a recorded result that actually supports the
claim, rather than merely an attempted test command. If the bank has only the six base tags,
set `positive_tags=("verified_result",)` and omit `changed_implementation` from the bank.
Questions and their definitions stay local configuration, not packaged data.

The pipeline then:

1. Reuses each tier's saved actor group and judges it under the frozen episode judge before
   point selection. A historical mechanical verifier pass remains a separate diagnostic.
2. Tags actor messages, tool calls with their results, and final answers step by step through
   the generic decision tagger. Context-only and experiment-imposed actions are excluded.
   Positive tags never enter the error-only bank, error totals or actor reward.
3. Forms same-study pass/fail pairs under that one judge. A bounded seeded sample prioritizes
   failed attempts that the historical verifier passed. No pair crosses a task, starting
   checkpoint, world, actor policy or tier. Unassessable episodes remain coverage gaps.
4. Asks an LLM to align contiguous windows around the corresponding decision. The windows may
   have different indexes; the full ordered episodes accompany the capped step views.
5. Computes the poor attempt minus the good attempt: Added and Removed tags. A tag must be
   confidently present on one side and confidently absent on the other. An unresolved
   comparison remains uncertain and does not enter an accepted group. With the default
   confidence, 0.3 and 0.7 also remain uncertain.
6. Asks a separate LLM draw to audit the mechanism, both windows and event citations. Legitimate
   test corrections, valid alternatives, recoveries and missing evidence must be considered.
   Simulator failures and missing tools need research rather than an actor-error rubric.
7. Saves accepted pairs with the poor-side checkpoint before the action, then groups identical
   directional Added/Removed signatures. Group identities include the exact bank digest, so
   equivalent signatures can be joined across saved reports without pooling their outcomes.
   Each report retains its task, world, tier, judge, source episodes and individual pair evidence.
8. Registers a compact `behavior_differences` reference for selectors and critics, and carries
   the exact signatures, paired steps, event citations and checkpoints into the family
   authors' coalition. The world and rubric authors remain independent.

For example:

```text
Good: changed_implementation, verified_result
Poor: modified_evaluation, unsupported_success_claim

Added:   modified_evaluation, unsupported_success_claim
Removed: changed_implementation, verified_result
```

`run_behavior_differences` (`variance/behavior_pairs.py`) is also a Python operation over a saved
study and its saved judging. Its `behavior-differences` artifact contains `differences.json`:
tag matrices, full evidence, selected pairs, alignments, audits, accepted groups and failures.
`load_behavior_differences` verifies the pinned report, tag text bindings, judgments, computed
differences and pre-action checkpoints. Failed or malformed paid responses are retained; replay
does not redraw them. Alignment and audit calls run at bounded concurrency and are included,
with step tagging, in the pipeline's up-front cost inventory.

This is opt-in analysis, not a new reward function. Accepted differences are reviewed hypotheses,
not causal proof, task admission or Harbor qualification. If there is no judged pass/fail
contrast, or every pair is uncertain or refused, the saved report has no accepted groups.
This registration currently cannot be combined with `mining.schedule`; generate the behavioral
evidence first and explicitly register it in a frozen schedule's base references.


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
`traceml-labels` artifact is registered as a `traceml_labels` reference, step labels and moves
and, when an `error-points` artifact is registered as an `error_points` reference, each attempt's
points ranked by the error tags of the steps decided there (`read_error_points`, locators
`errors:<n>` and `errors:<n>/point:<k>`).
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
so the schedule resumes and replays from disk without a model call. A schedule given a nudge
curriculum (`TrainingScheduleSpec.curriculum`, a `CurriculumPlan` over the records stage's
curriculum) names the current strength tier's rungs in each plan, strongest first, until only the
un-nudged targets remain; without one its identity and plans are unchanged.

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

**Searched scenarios and the nudge ladder** (`variance/family_search.py`). With
`DecisionFamilySpec.search` set, step 1's scenario draw is replaced by a measured variant search
(`searched_scenarios`, over `search_variants` and `variant_measurer`): every round's set is
bound, witnessed, contrasted and measured with the joint measure, and only an admitted set becomes
the family's scenarios, whose later stages replay the measured variants. With
`DecisionFamilySpec.ladder` set, an admitted family builds a nudge ladder (`family_ladder`, over
`build_nudge_ladder` and `ladder_measurer`) over the measured state whose free agents least often
chose its correct option (`ladder_base`). Every curriculum rung (each admitted nudged rung,
strongest first, then always the un-nudged base) is recorded as its own `RungTask` with its
strength, ladder position, judge binding and replayed variant. Both hooks need the joint measure,
stay out of the draws identity while unset, and bind each scenario set through one
`FamilyBinder`: the caller's judge function is asked for a whole set at a time, before any of the
set's contrasts, so a caller can author one frozen world, rubric and judge per set.

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
success; `efficiency_diagnostic` and `behavior_diagnostic` are reported and never decide it. The
rubric author (`rubric-author-v4`) may write either diagnostic role, and a library entry decided
`diagnostic` is cited only by diagnostic criteria; a bad-behavior linter decided `diagnostic` is
cited only by `behavior_diagnostic` criteria (`behavior-library-v2`), which the judge reports but
never requires.

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
| `executable/trial.py` `_judged_result` for a task with `dense`, `harbor/client.py` `dense.json` | `judge_dense` over the same ledger evidence | A separate dense training signal; never changes `reward` |

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

## Dense error-mode reward

The dense reward is a training signal kept beside the outcome reward, never in place of it. The
frozen episode judge's verdict stays the only outcome reward (pass 1, fail 0, else unscored). The
dense signal says where, inside the actor's own actions, a tagged behavior was established as a
violation (a penalty) or as genuinely useful (a bounded credit).

### Frozen definition

`DenseRewardDefinition` (`variance/dense_reward.py`) is authored with the Harbor environment and
bound to the task as `ExecutableTask.dense` (a `DenseRewardJudge`: the definition plus admitted
request settings), so it is part of `ExecutableTask.profile()` and of the exported package's
`profile_sha256`. The package's `frozen-binding.json` also records
`dense_reward: {schema_version, definition_sha256}` and `task.toml` declares
`dense_reward = "dense-reward-v1"`. Tag definitions and grading instructions stay on the trusted
backend and never reach the actor.

Each `DenseTag` has a polarity (`penalty` or `credit`), an axis (`correctness`, scored, or
`efficiency`, diagnostic only), a severity (`decisive` only for a correctness penalty), a
magnitude in (0, 1], grading instructions, `acceptable_when` look-alikes, `sources` (error-tag
proposition ids and linter locators that propose it) and `obligation_required`. `DenseBounds`
caps credit per behavior and per episode; the definition refuses an episode cap at or above any
decisive magnitude.

### Agentic judging

`judge_dense` (`variance/dense_judge.py`) judges one `EpisodeEvidence` with every prompt variant
of the profile (`variance/dense_prompt.py`: obligation-first, chronological and defense stances by
default, built with `dense_profile`, an open-weight judge alias). Each variant plans, reads ranges
when the evidence exceeds its input budget, reports one finding per candidate and gets at most one
repair after host validation. Candidates (`variance/dense_candidates.py`) come from every frozen
tag (`definition_candidates`), from error tags (`error_tag_candidates` over an
`ErrorTraceTags` summary) and from behavior linters (`linter_candidates` over `BehaviorGuidance`);
they are hypotheses, and the judge decides `absent`, `acceptable`, `uncertain` or `established`.

Observing is not establishing. For an `obligation_required` tag such as `modified_evaluation`,
the host accepts `established` only with the cited obligation (what the task, tests or grader
required), the actor's cited change, `result_refs` and a `weakening` explanation; otherwise the
repair is requested and a still unsupported finding votes `uncertain`. Legitimate test
corrections and alternative valid solutions are `acceptable` (scored, reward 0).

Variants are reconciled by strict majority; without one the finding is `uncertain`. A malformed
or unavailable variant votes `uncertain`, so it can never help establish anything. Uncertain
findings are recorded and unscored. Efficiency findings are recorded as diagnostics and never
scored.

### Bounded aggregation

`aggregate` assigns rewards:

- only `established` correctness findings are nonzero; decided non-established ones are 0;
  uncertain and efficiency findings are None;
- a credit behavior counts once (its first finding), capped at `per_behavior_credit`; repeats
  earn 0, so verbosity, repeated checks or extra steps add nothing;
- total credit is scaled to at most `episode_credit` (strictly below every decisive magnitude);
- once a decisive violation is established every credit is zeroed, so the episode total is at or
  below minus the largest decisive magnitude;
- the episode penalty sums behaviors once each and is scaled so the total is never below -1;
  a behavior's bounded penalty is split equally across its retained findings;
- per behavior, a finding whose actor actions an earlier same-tag finding already covers is a
  duplicate and earns 0, so a behavior proposed by several candidates (definition, error tag,
  linter) is applied once, and the per-finding rewards always sum to the episode total;
- an established finding with no actor span (for example one citing only an experiment-imposed
  action or its reply) is unscored with an `unscored_reason`; host validation already refuses
  such citations, since observed refs must resolve to the actor's own scored actions or the
  replies to them.

`dense_reward_test.py` checks these properties over randomized definitions and finding lists.

### Spans: rendered versus sampled tokens

`actor_spans` (`variance/dense_spans.py`) maps each cited event to the actor's trajectory: a tool
call by `call_id`, a message or final answer by exact text, and a world reply to the call it
answers. Context events and experiment-imposed actions never map. Every `DenseSpan` carries
rendered coordinates: `turn_index`, `part` (`content` or `tool_call`), `tool_call_index`,
`call_id`, `char_start`/`char_end` within the rendered part text and that text's digest. Tool-call
text is `json.dumps(arguments, sort_keys=True)` (`tool-call-arguments-json-v1`).

Rendered offsets are not sampled-token positions. A span carries `sampled` offsets (and
`granularity: sampled_tokens`) only when the actor's actual sampled tokens were bound
(`SampledTokens`, from a provider response with token records or the trainer rollout, under a
named `binding`) and their decoded text reproduces the part text exactly. Otherwise the span is
`granularity: action`. A continuation branch indexes turns by model call; a Harbor ledger
(`trajectory_from_ledger`) has one turn per actor action, and the trainer realigns by `call_id`
or message text.

### Export schema `dense-reward-v1`

`DenseRewardExport` (`dense.json`): `schema_version`, `episode_id`, `evidence_sha256`,
`definition_sha256`, `status` (`judged` or `unavailable`), `outcome` (the separate outcome
reward: judge, assessment, verdict, reward), `findings` (every finding: `tag_id`,
`candidate_ids`, `decision`, `observed_refs`, `obligation`, `change`, `result_refs`, `weakening`,
`rationale`, `unscored_reason`, `quote`, `votes`, `axis`, `polarity`, `severity`, `scored`, `reward`, `spans`),
`episode` (`total`, `credit`, `penalty`, `decisive`, `decisive_magnitude`, `credit_capped`,
`penalty_scaled`, `scored`, `unscored`, `diagnostics`), `trajectory`, `tool_call_rendering`,
`sampled_tokens` (the bound binding or null), `granularity` (`sampled_tokens`, `action` or
`mixed` over rewarded spans), `trainable`, `variants` (each prompt variant's status and saved
draws) and `failure`. It is saved in a `dense-assessment` artifact with `inspection.json` (every
draw of every variant).

### Harbor

For a task with both `judge` and `dense`, `ExecutableHarborTrial.result` judges the closed
ledger's evidence for the outcome, then runs `judge_dense` over the same evidence with
`definition_candidates` and the ledger trajectory, and adds the export as `result["dense"]`
(`trainable` is true only when the outcome is rewarded). The verifier (`harbor/client.py`) writes
it to `dense.json` next to `reward.json`, which is unchanged. The verifier runs in its own
environment, so unscored findings are recorded and never shown to the actor.

### Training records and the trainer

`attach_dense` (`variance/dense_records.py`) fills `ContinuationGroup.dense` with one
`DenseEpisode` (findings, spans, episode total) per judged slot, after checking the dense
assessment judged exactly the evidence the slot's outcome assessment judged. Advantages are
unchanged.

The external GRPO trainer combines them as `combine_advantages` (`variance/dense_trainer.py`),
which is the executable specification. For a group of `G` episodes with outcome rewards `R_i`:

```text
A_i     = (R_i - mean_g R) / std_g R        population std over scored episodes; 0 when std = 0
                                            (group_advantages); unscored R_i masks every token
D_i,t   = sum of r_f over scored findings f whose placed range covers actor token t
d_i,t   = clip(beta * D_i,t, -kappa, +kappa)
A_i,t   = clip(A_i + d_i,t, -C, +C)
defaults: beta = 0.5, kappa = 0.5, C = 5
```

`r_f` is the export's bounded finding reward; the dense term is not group-normalized because it
is sparse. `token_rewards` places findings on the trainer's sampled token stream: a span is
token-local only when it carries `sampled` offsets under the same binding the trainer's rollout
declares; otherwise its credit falls back to the whole action (every sampled token of that
message or tool call). `EpisodeTokens` refuses token-local rewards without a declared
`sampled_binding`, so rendered offsets alone never become token-local credit.

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
mistake patterns) and step by step (error-ranked points, saved as each tier's `error-points`
artifact and registered as its selectors' and critics' `error_points` reference). Uncertain steps
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
- **Version 2 selector records.** The records stage scores every selector draw. With
  `records.point_decisions` (default on), every continued point of each tier's run gets its
  options derived from its judged continuations (`variance/point_options.py`): an agentic option
  proposer (`records.option_proposer`, default the decision author) draws once per stance
  (`action_clustering`, `outcome_contrast`, `commitment_tracing`), each option citing the
  continuation events that show it, and a reconciling draw merges two or more valid proposals;
  each judged continuation is then classified by the shared choice classifier
  (`classify_free_study`, `records.option_classifier`, default the family classifier) into a
  `PointDecision`. A point a family was measured from also carries that family's joint score
  (`family_decisions`), which decides its value; a point whose family was measured but not
  admitted scores zero. Each point is saved as a `point-options` artifact and replays without
  calls; the plan prices every proposer, reconcile and classifier draw. A family's eligibility names the `decision_variance` condition when the
  joint measure blocks it.
- **Variant search and the nudge ladder** (`variance/variant_search.py`,
  `variance/nudge_ladder.py`, wired by `variance/family_search.py`). On by default with the
  decision measure (`authoring.variant_search`, `authoring.nudge_ladder`): each family's
  scenarios come from up to `authoring.search_rounds` measured search rounds on
  `authoring.search_author` (default the scenario author), and each admitted family builds a
  ladder of up to `authoring.ladder_rounds` rounds with nudges of `authoring.ladder_strengths` on
  `authoring.nudge_author` (default the scenario author), leak- and realism-reviewed by
  `authoring.nudge_reviewer` (default the rubric author), all open-weight. Each search round's set
  and each ladder round's rungs get their own packet, world, rubric, freeze and judges, so every
  rung of one measured round shares one rubric. The plan prices every round at its ceiling
  (`variance/pipeline_ladder.py`). A re-registration under revised judges contrasts one scenario
  draw; the search and ladder of the first registration stay recorded on the family's outcome.
  The records stage turns the ladders of admitted families into one curriculum
  (`RecordsOutputs.curriculum`, from `nudge_curriculum`): admitted rungs from strong to weak,
  every un-nudged target last. A `TrainingScheduleSpec` given that curriculum as a
  `CurriculumPlan` names the current strength tier's rungs in every rollout plan
  (`RolloutPlan.curriculum`), moving to the next tier every `rollouts_per_tier` rollouts.
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

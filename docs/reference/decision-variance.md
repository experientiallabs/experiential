# Decision-variance environments

This pipeline turns agent traffic into long-horizon environments whose difficulty comes from
decisions that matter. A point in a trajectory counts as a decision only when competing actions
taken from that exact checkpoint, continued under one frozen policy and budget, lead to measurably
different outcomes. A family of environments qualifies only when which action is better flips
across coherent variants of the world. Everything here is a Python API; three stages are also
`exp eval` continuation actions.

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
2. `extract_rules_in_parallel` makes one draw per trajectory proposing preconditions, effects,
   observations, invariants (optionally as predicates in a closed language) and open questions.
   Answers citing anything outside their own trajectory are kept as failures, never redrawn.
3. `merge_rules` keeps contradictions and unknowns; a rule is supported only with two or more
   supporting trajectories and no contradiction. Witnesses (native or specification) attach by
   rule id.
4. The rule book is saved as Markdown (`README.md`, `tools/<tool>.md`, `environment.md`) plus
   `rules.json` in one `rule-book` artifact.
5. Properties are checked against simulated sessions and boundary workloads challenge numeric and
   membership boundaries; a contradicted rule with no independent witness keeps every family that
   relies on it diagnostic.

A bound rule book makes the world model a state machine: the model proposes each transition and
`check_proposed_transition` (`exp/simulation/rules/checker.py`) decides. A rule-breaking proposal is
re-asked with the broken rules as feedback and never committed. Durable sessions take an opt-in
`transition_checker`; executable worlds bind the rule book into `GroundedRuntimeDefinition`.

## Screening: judges, critics and continuations

`run_actor_critic` (`variance/actor_critic.py`, CLI action `actor-critic`) runs G attempts of a
mining-probe task, picks one trajectory by seed, lets J point selectors (`judges`) choose up to K
points and C critics estimate the success probability at each, then runs M continuations per
point. Point selectors, critics and the final episode judge are separate roles even when they
share a model. Judges and critics read a privileged packet (every attempt, its grade, the group
rate and the verifier) and both carry the eval-audit checklist. Critics are scored by squared
error against measured rates; a critic with nothing graded is unscored. The judges' spread score
is a screening heuristic only and never a training reward. Every measured rate records its
`EstimateMeaning` (actor, harness, budget, sampling, world, verifier).

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
`forced_response` (the default: one model call, no tools) or `read_only_tools`, which serves
those locators from frozen documents under an `EvidenceBudget` per draw (model calls, tool calls,
input and output tokens, wall clock, dollars). Nothing is executed and nothing reaches the actor.
`run_analysis` saves every request, tool result and answer as an `AnalysisTranscript` that
replays without calls. The CLI cost inventory is an upper bound: each selector and critic row is
priced at `budget.model_calls` requests at the role's attempt ceiling, the group at its own
ceiling and `J * K` suffix jobs of M continuations at the continuation ceiling.

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
   through the preregistered action-contrast experiments.
4. `family_flip`: at least two variants with separated, different winners that match their declared
   correct options; a contradicting winner goes to triage and blocks qualification.
5. Fresh confirmation on independently seeded samples, then rubric grading
   (`variance/grader.py`), which separates insufficient evidence, ambiguous criteria, disagreement,
   pass, fail and plain uncertainty.

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
condition; training records exist only for eligible families. `export_family_harbor` packages a
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
boundary) or `rejected`, with the start checkpoint proven to restore exactly.
`review_compatibility` records every disagreement between the world and the rubric without
reconciling them.

## Judge and evidence contracts

`JudgeProfile` (`variance/episodes.py`) fixes the judge model alias and snapshot, prompt version
and digest, output-schema digest, temperature, output allowance, reasoning effort,
`evidence_tool_budget` and `maximum_input_bytes` (default 128000, the shared author default).
`JudgeDefinition` adds the rubric artifact, `rubric_sha256` (the digest of the rubric file's
content, refused when it equals the artifact's manifest digest), the event contract
(`episode-events-v1`), the task digest and the world digest. Any change is a new definition and
new labels.

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
exactly by protocol, definition identity, evidence identity and input budget. The prompt and its
schema live in `variance/judge_prompt.py` (`JUDGE_PROMPT_VERSION`); `judge_profile` binds both.

Every place on the execution-to-reward path that derives a success bit, an empirical rate, a
critic target, an advantage, an eligibility result or a Harbor reward, classified as semantic
grading (migrates to the judge) or simulator and protocol validity (stays code). Paths are relative
to `exp/optimize/evaluation/`; function names are authoritative where line numbers move.

| Site (file:line) | Current source | Kind | Target |
|---|---|---|---|
| `continuations/study.py:687`, `:721` `_run_branch` | `grade(spec.verifier, outcome)` sets `ContinuationBranch.assessment` (`VerifierGrade`) for completed and failed branches | semantic | Build `EpisodeEvidence` from the branch's actor calls, steps, receipts and final state; `judge_episode`; store the `JudgedEpisode` reference. Branch `status` (completed, failed, outcome_unknown) and `actor_stop` stay protocol facts; an outcome-unknown branch is still unscored, an actor budget stop is still judged |
| `continuations/study.py:191` `known_outcomes` | `VerifierGrade.status == "pass"`, unassessable dropped | semantic aggregation | Labels from `episode_reward(load_assessment(...))`; unscored stays in the planned denominator |
| `continuations/study.py:259` `ContinuationStudy.metrics` | `resumed_success_rate` and `brier` against `predicted_success` from verifier labels | semantic | Same arithmetic over judged labels, reported with the judge definition identity |
| `continuations/action_contrast.py:177` `_assessment`, `:187` `grade_intervention_outcome` (`grade` at `:207`) | Re-grades the imposed action plus suffix with the task verifier | semantic | One `EpisodeEvidence` with the imposed action as `owner=experiment`, `scope=scored`, then the suffix, judged once; no separate regrade |
| `continuations/action_contrast.py:210` `_metrics` | Arm success bounds and pair differences from verifier labels; `goal_failures`/`policy_failures` from `VerifierGrade.goal/policy` | semantic | Bounds from judged labels; goal/policy counts replaced by counts of judged not_met required criteria |
| `variance/evidence.py:51` `group_rate`, `:125` attempt `grade`, `:147` `success_rate` | Verifier grades and group rate shown to point selectors and critics | semantic (historical) | Discovery packets label these historical; scored variant predictions read judged rates of the frozen variant judge |
| `variance/evidence.py:64` `candidate_branches`; `variance/actor_critic.py:336` `_pick` | Completed attempts with checkpoints and enough remaining calls | protocol | Unchanged; record the eligible population as selected, not an unconditional draw |
| `variance/actor_critic.py:465`-`:484` `_point_result` | `known_outcomes` and per-branch 1/0/None from `VerifierGrade` feed `PointOutcome` and `group_advantages` | semantic | `episode_reward` of each continuation's assessment |
| `variance/actor_critic.py:489` `_reward_judge`, `:498` `_reward_critic`, `:526` `_ineligible`, `:550` `_rank` | Selector spread score, critic negative Brier, ranking over `PointOutcome` rates | consumer of semantic rates | Unchanged arithmetic over judged rates; the spread score stays screening only |
| `variance/rewards.py:125` `critic_score`, `:158` `judge_score`, `:182` `group_advantages` | Pure arithmetic over supplied rates and rewards | encoding | Unchanged; inputs come only from judged assessments |
| `variance/exposure.py:286` `estimate_meaning` (`verifier_sha256` at `:308`) | Estimate meaning binds the verifier digest | identity | Bind the `JudgeDefinition.identity` instead, so a new rubric or profile is a new estimate |
| `variance/contrast.py:742` `option_outcome` | Option counts, Wilson and separation bounds, goal/policy failures from cell `VerifierGrade` | semantic | Judged labels of each forced-action cell |
| `variance/contrast.py:835` `variant_verdict`, `:877` `family_flip` | Separated leader from option bounds, compared with the declared `correct_option` | consumer of semantic rates; the declared option is an authoring hypothesis | Bounds from judged cells; a mismatch with the declared option stays a triage finding, never a reward |
| `variance/contrast.py:452` `_verdict` (witness) | Executed witness calls show each deciding fact, no error, no verbatim answer, enough calls | protocol | Unchanged; it proves discoverability, not success |
| `variance/milestone.py:650`-`:673`, `:780` | Point separation from `run_option_contrast` | consumer of semantic rates | Follows judged contrasts; no separate grading |
| `variance/family_stages.py:255` `grade_variants` (`variance/grader.py:281` `grade_episodes`) | Probabilistic criterion grading of confirmation episodes; only ambiguity and grading success reach eligibility | semantic diagnostic | Judge every confirmation episode; eligibility counts judged, unscored, insufficient-evidence and rubric-problem outcomes separately; criterion probabilities stay calibration evidence only |
| `variance/records.py:424`, `:432` `family_flip`, `fresh_confirmation` conditions | From `FlipVerdict` over verifier-labelled contrasts | semantic | From judged contrasts of the frozen variant judge |
| `variance/records.py:450` `critic_scores` | Critic draws without a scored reward | consumer of semantic rates | Unchanged over judged rates |
| `variance/records.py:478` `rubric_unambiguous` | `rubric_graded` and ambiguous criteria from `grade_variants` | semantic | Replaced by judged-coverage and rubric-problem conditions; one uncertain criterion is not proof of ambiguity |
| `variance/records.py:443`, `:457`, `:464`, `:471` (dispositions, tool coverage, rule witnesses, policy consistency) | Audit and world evidence | protocol | Unchanged |
| `variance/records.py:564` `training_records` (`measured_rate` `:604`, `advantages` `:622`) | Copies critic targets and group advantages from the discovery `ActorCriticReport` | semantic handoff | Build from the family's judged variant studies under the frozen variant judge; discovery scores stay diagnostic |
| `variance/family.py:598`, `:617`, `:621`, `:682` `_continue`/`_finish` | Selection, confirmation, rubric grading and records wiring | consumer | Freeze the `JudgeDefinition` per variant before scored critic predictions and fresh continuations; pass judged studies to records |
| `variance/family.py:357` `export_family_harbor` (`:396`) | Qualified only when eligible, confirmed and an executable qualification exists | consumer | Package the exact private judge binding; qualification of the LLM assessment path is separate from observable-contract qualification |
| `executable/trial.py:231` `result` (`grade` `:245`, `reward` `:258`) | Harbor reward from `grade(task.job.study.verifier, outcome)` when qualified | semantic | `judge_episode` over the trial's harness-recorded evidence, including captured intermediate dialogue; reward `episode_reward` |
| `harbor/client.py:164`-`:165` | Writes `reward.json` from the backend's `reward` and `status` | encoding | Unchanged; it only serializes the judged reward |
| `harbor/trial.py:304` `result` (`grade` `:310`, `task_decision` `:365`) | Mechanical grade combined with dual-evaluator axes for `EvaluationTask` Harbor trials | separate evaluation mode | Keeps its explicit contract; never used for decision-variance family exports |

## What this does not establish

Structural gates are necessary, not sufficient. Simulator continuations reduce sampling noise but
not simulator bias, so boundaries without independent witnesses stay diagnostic. Executable
qualification does not cover native fidelity, rubric judging, tier separation or containment.

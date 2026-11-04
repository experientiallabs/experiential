# Decision-variance environments

This pipeline turns agent traffic into long-horizon environments whose difficulty comes from
decisions that matter. A point in a trajectory counts as a decision only when competing actions
taken from that exact checkpoint, continued under one frozen policy and budget, lead to measurably
different outcomes. A family of environments qualifies only when which action is better flips
across coherent variants of the world. Everything here is a Python API; two stages are also
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
mining-probe task, picks one trajectory by seed, lets J judges choose up to K points and C critics
estimate the success probability at each, then runs M continuations per point. Judges and critics
read a privileged packet (every attempt, its grade, the group rate and the verifier) and both carry
the eval-audit checklist. Critics are scored by squared error against measured rates; a critic with
nothing graded is unscored. The judges' spread score is a screening heuristic only and never a
training reward. Every measured rate records its `EstimateMeaning` (actor, harness, budget,
sampling, world, verifier).

## Decision families

`run_decision_family` (`variance/family.py`) takes one screened point through:

1. Decision-state and scenario authors (`variance/authors.py`): options executable as competing
   actions, variation axes with permitted values grounded in rules or observations, invariants,
   single-proposition criteria. Scenario sets must pass `check_scenarios` (option identities,
   permitted values, invariants compared before and after, distinct states, frozen instruction,
   an observation witness for every deciding fact within the remaining calls).
2. Coherent variants (`variance/variants.py`): fork before the first turn that revealed an edited
   fact, edit grounded world state there, check public context, instance facts and clock or pending
   events, and prove restoration.
3. Executed witnesses and option contrasts (`variance/contrast.py`): the witness calls must show
   every deciding value without stating the answer; each option is forced from the same checkpoint
   through the preregistered action-contrast experiments.
4. `family_flip`: at least two variants with separated, different winners that match their declared
   correct options; a contradicting winner goes to triage and blocks qualification.
5. Fresh confirmation on independently seeded samples, then rubric grading
   (`variance/grader.py`), which separates insufficient evidence, ambiguous criteria, disagreement,
   pass, fail and plain uncertainty.

`TrainingEligibility` (`variance/records.py`) is computed, never asserted, and lists every failed
condition; training records exist only for eligible families. `export_family_harbor` packages a
confirmed variant for Harbor, diagnostic unless the family is eligible and an executable
qualification is supplied.

## Milestone comparison

`run_selection_comparison` (`variance/milestone.py`) preregisters judge-chosen points against
uniform, late-failure and midpoint baselines at an equal point budget and counts how many turn out
to be decisions. `compare` reports `judge_better` only when the paired lower bound beats every
baseline; otherwise `not_shown`.

## What this does not establish

Structural gates are necessary, not sufficient. Simulator continuations reduce sampling noise but
not simulator bias, so boundaries without independent witnesses stay diagnostic. Executable
qualification does not cover native fidelity, rubric judging, tier separation or containment.

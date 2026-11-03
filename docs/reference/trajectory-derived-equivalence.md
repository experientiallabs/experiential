# Trajectory-derived actor equivalence

`compare-environments` asks a semantic judge to rate each paired execution on five reporting
dimensions: goal, decisions, facts, consequences and completion. The judge rates under an
explicitly reviewed, frozen environment rubric derived from admitted **development
trajectories**. Evaluation trajectories never write its criteria. The descriptions, score
anchors, valid variations and uncertainty conditions are specific to the observed environment.

The rubric shapes semantic ratings only. Trajectory admission stays with the registered
[admission policy](admission-policy.md): its six mechanical dimensions, gates and margins decide
whether a trajectory is admitted, and judge ratings never admit, refuse or calibrate anything.

`exp.optimize.evaluation.equivalence.rubric` owns this lifecycle:

1. Register a `RubricTrajectory` (`register_rubric_trajectory`) containing its `TaskCase` and
   complete `WorkflowEvidence`, with exact source artifact inputs. Trace-backed tasks must
   resolve to fit sources in the frozen build. Authored development tasks retain their declared
   provenance limits.
2. Classify the exact source artifacts and their complete ancestry with the diagnostic source
   admission owner (`declare_source_access`). Unknown or protected dependencies refuse
   disclosure.
3. Submit a `RubricDerivationSpec` through `exp eval PROJECT --continuations FILE`, with action
   `derive-equivalence-rubric` and payload field `rubric_derivation`. It freezes the proposer
   identity, catalog digest, state coverage, finite physical attempt ceiling, input bound and
   deadline. `--dry-run` validates sources and the proposer without a call; the one proposal
   request needs spend consent; a saved draft replays without consent or calls.
4. Inspect the saved draft. Every criterion must cite an exact excerpt from an allowed public
   trajectory field. Inspect whether the excerpt supports the criterion, its score meanings and
   proposed alternatives. Citation validation proves the excerpt exists, not that the criterion
   follows from it.
5. Submit action `freeze-equivalence-rubric` with a `rubric_review` payload naming the exact
   draft artifact, reviewer and acceptance rationale. The optional `proposal` field records
   explicit author corrections, which must retain valid citations and declared state coverage.
   All anchors must compare the two executions: matching supported mistakes can be faithful
   while competence and safety findings remain adverse. This is an explicit author review;
   independent judge calibration requires separate reviewed labels.
6. Set the printed reference as the required `rubric` field of `EquivalenceSpec`. The comparison
   registration cites it before any execution, so the rubric is fixed before the outcomes it
   rates. Development families must be disjoint from evaluation families: with a pinned
   `policy`, lineages are grouped by its declared `families`, so a rubric derived from one
   lineage of a family cannot rate another lineage of it. Source admission is rechecked before
   every review disclosure, including after a rubric is frozen.

Derivation exposes public task context, actual actions, replies, stop reason and final answer.
It excludes private state probes, audit metadata and verifier labels. The proposer is instructed
to extract action dependencies, argument roles, observed error classes and recovery, stopping,
policy constraints and harmful extra actions. Co-occurrence alone does not prove causality.
Requests that exceed their bound are refused rather than silently truncated. The equivalence
review transport preserves requests, responses, failures and attempt accounting; a failed or
unknown dispatch does not buy another draw implicitly.

## Measuring incomplete state

With partial or unknown state, compare observable decision-relevant relationships: how an actor
uses tool results, responds to errors, recovers and stops. Equivalent supported alternatives need
not use identical tool sequences. Different errors are not equivalent merely because both runs
fail. Inventing data can change a later decision and remains a behavioral mismatch.

A pair is `behavior_assessable`, and judged, when both executions completed with only valid
steps from the same first actor request and every declared initial-state probe agrees. A known
candidate failure is a material difference with no behavior to rate. Missing final-state probes
leave material outcomes unknown while behavior stays assessable. Declared initial and final probe
paths are optional unless the rubric declares full state coverage, which requires both in every
case.

Reports retain, per condition and per case, behavior-assessable pair counts and each criterion's
assessed, unassessable, judge-failure and unjudged counts, plus its 1 to 5 mean with repeats
averaged within each case first. That mean is not an accuracy percentage and carries no interval.
Unknown goal outcomes remain unassessable even if the judge awards them full marks; the overall
normalized score is absent when any criterion is unavailable. Material agreement and
task-result matrices remain separate. Matching failures can be faithful simulation without
demonstrating competence or qualifying training samples.

Fixed-action reply replay remains a different diagnostic. Its exact-match rates do not measure
actor equivalence because the actor did not choose the subsequent actions in each environment.

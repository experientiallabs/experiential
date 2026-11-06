"""Protocol versions stand in for source text in saved-result identities; this guards them.

Calibration, the actor council, task refinement, prompt hill climbing, private workbook views, the
restorable environment backends and local process containment bind their saved results to a
protocol constant instead of hashing their own source. Saved results then survive moves,
formatting and unrelated edits, and re-key only when the protocol is bumped. Provider attempt
timeouts follow the same rule: each admitted physical attempt records the protocol of the
derivation that bounded it.
Calibration, the actor council, task refinement, private workbook views, workbook instance facts
and the restorable environment backends bind their saved results to a protocol constant instead
of hashing their own source. Saved results then survive moves, formatting and unrelated edits,
and re-key only when the protocol is bumped. Provider attempt timeouts follow the same rule:
each admitted physical attempt records the protocol of the derivation that bounded it.

This test keeps that honest. Each protocol lists the source that decides its behavior, with that
source's digest at the protocol's current value. When the covered source changes, the test fails
until one of two things happens:

- behavior changed: bump the protocol constant (saved results re-key) and record the new digest;
- pure refactor (a move, imports, formatting, comments): record the new digest, keep the constant.

Either way the choice is explicit in review instead of an accident of file layout.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from types import FunctionType, ModuleType

import pytest

Source = ModuleType | type | FunctionType  # what the covered references resolve to


@dataclass(frozen=True)
class Protocol:
    """One protocol constant, the source it covers and that source's recorded digest."""

    constant: str  # "module:NAME"
    value: str
    covers: tuple[str, ...]  # "module" (whole file) or "module:qualname" (one object)
    digest: str


PROTOCOLS = (
    Protocol(
        "exp.optimize.evaluation.variance.family_scopes:PROBE_PLACEMENT_PROTOCOL",
        "probe-placement-v1",
        (
            "exp.optimize.evaluation.variance.family_scopes",
            "exp.optimize.evaluation.variance.build_records:ProbePlacement",
            "exp.optimize.evaluation.variance.build_records:ProbePlacementRegistration",
            "exp.optimize.evaluation.variance.build_records:FamilyScoping",
            "exp.simulation.mining.environment_families:EnvironmentFamilySpec",
            "exp.simulation.rules.contracts:ToolSchemaCoverage",
            "exp.simulation.rules.contracts:tool_schema_coverage",
            "exp.optimize.evaluation.variance.bootstrap_selection:select_probe_task_bases",
        ),
        "93670037f5970400afeaf725eb4f3835dedacc347e8c74539f076561aee4d81b",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.environment_catalog:ENVIRONMENT_CATALOG_PROTOCOL",
        "environment-catalog-v2",
        (
            "exp.optimize.evaluation.variance.environment_catalog",
            "exp.simulation.mining.tag_selection",
            "exp.simulation.rules.contracts:tool_schema_coverage",
            "exp.simulation.rules.contracts:ToolSchemaCoverage",
        ),
        "b8e7d2f2bf9511c504b626a593b80ec832cc0e45229bef85d2fb985cecc14172",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.behavior_differences:BEHAVIOR_DIFFERENCES_PROTOCOL",
        "behavior-differences-v1",
        (
            "exp.optimize.evaluation.variance.behavior_differences",
            "exp.optimize.evaluation.variance.behavior_pairs",
            "exp.optimize.evaluation.variance.pipeline_behaviors",
        ),
        "d1b011791ca26ab487e40d63bad8a9120a04fdce89b6f6f16a568365095198a2",
    ),
    Protocol(
        "exp.simulation.mining.descriptors:DESCRIPTOR_EMBEDDING",
        "hashing-descriptor-v3",
        (
            "exp.simulation.mining.descriptors",
            "exp.simulation.mining.lineage",
        ),
        "6f73c03012822be35706a703429b1c9b2941321e5c0c97ad1528db6464c7786d",
    ),
    Protocol(
        "exp.optimize.evaluation.tasks.calibration:CALIBRATION_PROTOCOL",
        "task-calibration-v12",
        (
            "exp.optimize.evaluation.tasks.calibration",
            "exp.optimize.evaluation.tasks.admission",
            "exp.simulation.evaluation.calibration",
            "exp.simulation.evaluation.tasks",
            "exp.simulation.evaluation.admission",
        ),
        "f15a5289e76ffa42f8363e0d85d7ed660873740886bf4a149c7ca832e433740d",
    ),
    Protocol(
        "exp.optimize.evaluation.tasks.challenges:CHALLENGE_PROTOCOL",
        "judge-challenge-v8",
        (
            "exp.optimize.evaluation.tasks.challenges",
            "exp.optimize.evaluation.tasks.challenge_runs",
            "exp.simulation.evaluation.calibration:task_decision",
            "exp.simulation.evaluation.calibration:blinded_case",
        ),
        "00191aa5ee5110a96da03c9fad4e0e789044b5468931b9dec62bb5d577241b62",
    ),
    Protocol(
        "exp.optimize.evaluation.council.runner:COUNCIL_HARNESS_PROTOCOL",
        "council-harness-v6",
        (
            "exp.runtime.agents.chat:ChatAgentRuntime",
            "exp.runtime.agents.pi:PiAgentRuntime",
            "exp.simulation.actor:_RecordingEnvironment",
            "exp.simulation.actor:run_actor_workflow",
            "exp.simulation.world_model.grounded_session:open_grounded_session",
        ),
        "185fec9723a9df3dc5e482ddbc598418b5112559a1262383d39e430fd35e3749",
    ),
    Protocol(
        "exp.optimize.evaluation.tasks.refinement:REFINEMENT_REQUEST_PROTOCOL",
        "task-refinement-request-v1",
        ("exp.optimize.evaluation.tasks.refinement:_request",),
        "7d532742e7fcd847233cf7af439fbb476d9dc5fde965ebee1ebf407fa4c70fa8",
    ),
    Protocol(
        "exp.optimize.evaluation.continuations.proposals:SELECTOR_REQUEST_PROTOCOL",
        "checkpoint-selector-request-v2",
        (
            "exp.optimize.evaluation.continuations.proposals:_canonical",
            "exp.optimize.evaluation.continuations.proposals:_shared_values",
            "exp.optimize.evaluation.continuations.proposals:_packet",
            "exp.optimize.evaluation.continuations.proposals:_render",
        ),
        "9a6f924c6f413f3da8911eeb078b3d45b92a89b9838766fbd14eaff378a54135",
    ),
    Protocol(
        "exp.optimize.evaluation.learning.sample_count:SAMPLE_COUNT_PROTOCOL",
        "sample-count-v4",
        ("exp.optimize.evaluation.learning.sample_count",),
        "1ada99fdb976a5f5c9bc1bc54c119b6f5f8453b87c0c3cd2f4b81aca30bd33c0",
    ),
    Protocol(
        "exp.optimize.evaluation.learning.agent_memory:AGENT_MEMORY_PROTOCOL",
        "agent-memory-v2",
        (
            "exp.optimize.evaluation.learning.agent_memory_requests",
            "exp.optimize.evaluation.learning.agent_memory:extract_candidates",
            "exp.optimize.evaluation.learning.agent_memory:consolidate",
            "exp.optimize.evaluation.learning.agent_memory:_merge_kind",
            "exp.optimize.evaluation.learning.agent_memory:_merge_chunk",
            "exp.optimize.evaluation.learning.agent_memory:_screen",
            "exp.optimize.evaluation.learning.agent_memory:_shown_candidate",
            "exp.optimize.evaluation.learning.agent_memory:_shown_lesson",
            "exp.optimize.evaluation.learning.agent_memory:_accepted_shown",
            "exp.optimize.evaluation.learning.screening:screening_score",
        ),
        "311e91099d43a0dc9c681881446cd988b8dc571e30a8c026117fbad10f6bb7e9",
    ),
    Protocol(
        "exp.optimize.evaluation.learning.screening:SCREENING_PROTOCOL",
        "learning-miss-screening-v1",
        (
            "exp.optimize.evaluation.learning.screening:screening_request",
            "exp.optimize.evaluation.learning.screening:_bounded",
            "exp.optimize.evaluation.learning.screening:_probabilities",
        ),
        "de36bab2fa626f87e9673bb8bfd220fa6e2051631cb443c15eb519016a353485",
    ),
    Protocol(
        "exp.optimize.evaluation.learning.lesson_evolution:EVOLUTION_PROTOCOL",
        "lesson-evolution-v3",
        (
            "exp.optimize.evaluation.learning.lesson_evolution:_attempt",
            "exp.optimize.evaluation.learning.lesson_evolution:evolve_lessons",
        ),
        "864ccb657a7e51ff10ce6e9f961aa2059ab2e377dc5de41234408cbe95d2dfdc",
    ),
    Protocol(
        "exp.optimize.evaluation.hillclimb.design:HILL_CLIMB_PROTOCOL",
        "hill-climb-v3",
        (
            "exp.optimize.evaluation.hillclimb.design:climb_split",
            "exp.optimize.evaluation.hillclimb.measurement",
            "exp.optimize.evaluation.hillclimb.stages",
            "exp.optimize.evaluation.hillclimb.rounds:run_climb_round",
        ),
        "b65f8761630693a3491f1dc1863cb697205f5b8d57050be862e2dcb7bb9dd0cd",
    ),
    Protocol(
        "exp.optimize.evaluation.hillclimb.proposals:EDIT_PROPOSAL_PROTOCOL",
        "prompt-edit-proposal-v1",
        (
            "exp.optimize.evaluation.hillclimb.proposals:train_evidence",
            "exp.optimize.evaluation.hillclimb.proposals:proposal_request",
            "exp.optimize.evaluation.hillclimb.proposals:_validate",
        ),
        "8b96977f5e07a23d8466599ca270d36d70a767474aa2f508853961e3ca03b172",
    ),
    Protocol(
        "exp.runtime.models.providers.attempt_timeouts:ATTEMPT_TIMEOUT_PROTOCOL",
        "provider-attempt-timeout-v2",
        ("exp.runtime.models.providers.attempt_timeouts",),
        "ce424e2025444d35f339df9c23da581a49102d27efd35a0a58ec6fccfa841fe2",
    ),
    Protocol(
        "exp.runtime.environments.workbook_edits:XLSX_ROW_EDITS_PROTOCOL",
        "xlsx-row-edits-v3",
        ("exp.runtime.environments.workbook_edits",),
        "977a23f451e187c67db9052c5eaa226a49b433727821a904d3708ea648ff6242",
    ),
    Protocol(
        "exp.runtime.environments.workbook_facts:WORKBOOK_FACTS_PROTOCOL",
        "xlsx-instance-facts-v1",
        ("exp.runtime.environments.facts_derivers", "exp.runtime.environments.workbook_facts"),
        "91377a65f4925f490e8196c206140ab96837223ef71472d2f3ff91d4ee1034d9",
    ),
    Protocol(
        "exp.runtime.environments.artifact_world:ARTIFACT_WORLD_PROTOCOL",
        "artifact-world-v2",
        ("exp.runtime.environments.artifact_world",),
        "46a47face27b447433d024c989da577496bbbed3e66eda502812a08698b5c495",
    ),
    Protocol(
        "exp.optimize.evaluation.executable.workbook:PRIVATE_VIEWS_PROTOCOL",
        "private-workbook-views-v2",
        ("exp.optimize.evaluation.executable.workbook",),
        "e14d471c25cadda4e5cc1750bf8784a8f3576d87b37589c8d70a4561a0ed8881",
    ),
    Protocol(
        "exp.runtime.environments.containment.linux:LINUX_CONTAINMENT_PROTOCOL",
        "linux-namespaces-v3",
        (
            "exp.runtime.environments.containment.linux",
            "exp.runtime.environments.containment.linux_gate",
        ),
        "f84fdea9abeb821d9203c4593d215a0f853e6757d7ac117f7ad23966ff930776",
    ),
    Protocol(
        "exp.runtime.environments.containment.darwin:DARWIN_CONTAINMENT_PROTOCOL",
        "darwin-kqueue-v1",
        (
            "exp.runtime.environments.containment.darwin",
            "exp.runtime.environments.containment.darwin_gate",
        ),
        "693bb76a1249204e492feaca8ae8508ed914ade3334792a6b008048dfcbf0375",
    ),
    Protocol(
        "exp.runtime.agents.cascade:CASCADE_VERSION",
        "decision-cascade-v3",
        (
            "exp.runtime.agents.cascade:action_key",
            "exp.runtime.agents.cascade:action_projection",
            "exp.runtime.agents.cascade:decision_candidates",
            "exp.runtime.agents.cascade:delegates",
            "exp.runtime.agents.cascade:CascadeModelClient",
            "exp.runtime.agents.cascade:admitted_draft",
        ),
        "6b6acd94daf56c93c0056fae980169889d050ffb6b49f518f81255f9052a9d2d",
    ),
    Protocol(
        "exp.optimize.evaluation.analysis.decision_cascade:CASCADE_FIT_PROTOCOL",
        "decision-cascade-fit-v6",
        (
            "exp.optimize.evaluation.analysis.decision_cascade:CascadeObjective.non_inferior",
            "exp.optimize.evaluation.analysis.decision_cascade:estimate_threshold",
            "exp.optimize.evaluation.analysis.decision_cascade:refused_drafts",
            "exp.optimize.evaluation.analysis.decision_cascade:select_threshold",
            "exp.optimize.evaluation.analysis.decision_points:DecisionPointSet.labels",
            "exp.optimize.evaluation.analysis.decision_points:state_request",
        ),
        "3c1a106e375bd744bbb42a0a0351d6d1b9b0c62e8c02c49209b2f03b84a028af",
    ),
    Protocol(
        "exp.runtime.agents.cascade_deciders:DECIDER_PROTOCOL",
        "cascade-decider-v1",
        (
            "exp.runtime.agents.cascade_deciders:next_action_request",
            "exp.runtime.agents.cascade_deciders:next_action_answer",
            "exp.runtime.agents.cascade_deciders:SampledDraftDecider.decide",
        ),
        "0adbb5717f465ad401271fc664c2270658049130acec59e248c986caaf6d7651",
    ),
    Protocol(
        "exp.runtime.environments.record_world:RECORD_WORLD_PROTOCOL",
        "sqlite-record-world-v1",
        ("exp.runtime.environments.record_world",),
        "8fb247418212f90d8963088e1323a0b4581dbaf937fc8cda590d784a31867b99",
    ),
    Protocol(
        "exp.simulation.induction.runner:RUNNER_PROTOCOL",
        "induced-program-runner-v7",
        ("exp.simulation.induction.runner", "exp.simulation.induction.sandbox"),
        "6fe35ac98d15b2722319ef70c22d9b9998be01c8a88492a16417242bc3528b1a",
    ),
    Protocol(
        "exp.simulation.induction.backend:INDUCED_WORLD_PROTOCOL",
        "induced-world-v1",
        (
            "exp.simulation.induction.backend",
            "exp.simulation.induction.contracts:ProgramDomain",
            "exp.simulation.induction.validation:induced_partition",
        ),
        "9b738d7a815e3b23db372a80bb2c8f848a14c2d7d768b88e3b7d2bc4cca70f4e",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.roles:ROLE_PROTOCOL",
        "variance-roles-v9",
        (
            "exp.optimize.evaluation.variance.roles",
            "exp.optimize.evaluation.variance.contracts:EstimationPoint",
            "exp.optimize.evaluation.variance.contracts:JudgeSelection",
            "exp.optimize.evaluation.variance.contracts:AuditFlag",
            "exp.optimize.evaluation.variance.contracts:CriticEstimate",
            "exp.optimize.evaluation.variance.contracts:CriticAssessment",
            "exp.optimize.evaluation.variance.evidence_tools",
            "exp.optimize.evaluation.variance.role_prompts",
            "exp.optimize.evaluation.variance.evidence",
        ),
        "6d677f3a0a1efed976abc83ea7d80d92987e3be198eddb455336665d37d8205e",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.extraction:EXTRACTION_PROTOCOL",
        "rule-extraction-v1",
        (
            "exp.optimize.evaluation.rules.extraction",
            "exp.simulation.rules.contracts:trajectory_view",
            "exp.simulation.rules.model",
        ),
        "794dbbff4b6eb696efe3295910d93537cba799e3062a3b9428bfd65a049711d5",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.agent_loop:AGENT_LOOP_PROTOCOL",
        "rules-agent-loop-v5",
        (
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.optimize.evaluation.rules.evidence_pages",
            "exp.optimize.evaluation.factory.authoring:author_request_bytes",
            "exp.optimize.evaluation.factory.authoring:_effective_request",
        ),
        "f920294eb96eb0b3ea93cbf955dd823a7a448f2faade181fa36d7ea5ef00685c",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.research:RESEARCH_PROTOCOL",
        "rule-research-v6",
        (
            "exp.optimize.evaluation.rules.research",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.simulation.rules.contracts:trajectory_view",
            "exp.simulation.rules.model",
            "exp.optimize.evaluation.rules.evidence_pages",
            "exp.optimize.evaluation.factory.authoring:author_request_bytes",
            "exp.optimize.evaluation.factory.authoring:_effective_request",
            "exp.optimize.evaluation.rules.direct_research",
        ),
        "457e5cd8a2d8a31197cbe2edcc20bc6b6dbf3c0135c5f3727f0753012ec8999f",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.weighing:WEIGHING_PROTOCOL",
        "rule-weighing-v5",
        (
            "exp.optimize.evaluation.rules.weighing",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.simulation.rules.merge:check_weighing",
            "exp.simulation.rules.model",
            "exp.optimize.evaluation.rules.evidence_pages",
            "exp.optimize.evaluation.factory.authoring:author_request_bytes",
            "exp.optimize.evaluation.factory.authoring:_effective_request",
        ),
        "db37de6a5a00b3295eda7105f894e2a01536fb2cf8c69bea02c1e1ab4830bca1",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.condense:CONDENSATION_PROTOCOL",
        "rule-condensation-v6",
        (
            "exp.optimize.evaluation.rules.condense",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.optimize.evaluation.rules.evidence_pages",
            "exp.optimize.evaluation.factory.authoring:author_request_bytes",
            "exp.optimize.evaluation.factory.authoring:_effective_request",
        ),
        "51bfd50e6abcf5bdd1730d7b10d41207b22317808d7a1d5bb950189eed002f2a",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.authors:AUTHORS_PROTOCOL",
        "variance-authors-v7",
        (
            "exp.optimize.evaluation.variance.authors",
            "exp.optimize.evaluation.variance.contracts",
            "exp.optimize.evaluation.variance.coalition",
        ),
        "6910fc1b9046dfea05b195e235fb0ecbe008fce4b69c317d0c13597af6cc3c64",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.briefing_review:BRIEFING_REVIEW_PROTOCOL",
        "briefing-leak-review-v2",
        ("exp.optimize.evaluation.variance.briefing_review",),
        "24f25452817cdd31b12059bcab887fae42b13ea8174e0ab0ed01aa4d240130e6",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.grader:GRADER_PROTOCOL",
        "variance-grader-v1",
        ("exp.optimize.evaluation.variance.grader",),
        "5581d2240bfa5181f7eb71729c0de7e2d3e5deed6f816b29f01bc9ffe070058e",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.judge:JUDGE_PROTOCOL",
        "episode-judge-v3",
        (
            "exp.optimize.evaluation.variance.judge",
            "exp.optimize.evaluation.variance.judge_prompt",
        ),
        "4504f963b8caaa9442d4c5a860f1d43793e4870428ec3e902cbacb2273f6a4e8",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.judged:JUDGED_LABELS_PROTOCOL",
        "judged-labels-v2",
        (
            "exp.optimize.evaluation.variance.judged",
            "exp.optimize.evaluation.variance.episode_evidence",
        ),
        "2b26ae0c1cda7b2ceaa1ab4cce7d8535cb638d99ff24eee9d41b96dd185b701c",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.admission:FAMILY_ADMISSION_POLICY",
        "family-admission-v2",
        ("exp.optimize.evaluation.variance.admission",),
        "db87ef56d6f53319d2c606685e674d5c58e544da8a3028d4fd9df7aa027e6e94",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.authoring_packet:AUTHORING_PACKET_PROTOCOL",
        "authoring-packet-v4",
        (
            "exp.optimize.evaluation.variance.authoring_packet",
            "exp.optimize.evaluation.variance.coalition",
        ),
        "2448351352aa1fc736472c3e7bf27cc67c50f298d8fb07afea62befbe17d9f07",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.world_author:WORLD_AUTHOR_PROTOCOL",
        "world-author-v2",
        ("exp.optimize.evaluation.variance.world_author",),
        "20303596f7bb5357aee4c624b2b94b3a6c6d59c297686534bd891d03a88051cb",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rubric_author:RUBRIC_AUTHOR_PROTOCOL",
        "rubric-author-v4",
        ("exp.optimize.evaluation.variance.rubric_author",),
        "3c8e830b11ef86d1f50a71b2a12f15b5c3b2d626b92bec4fad94f6b9b799794d",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rubric_agent:RUBRIC_AGENT_PROTOCOL",
        "rubric-agent-v2",
        ("exp.optimize.evaluation.variance.rubric_agent",),
        "87587e4a0e92ace340b6c48b51ce0621659c7e61ca8b634506a19dc34505118e",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rubric_revision:RUBRIC_REVISION_PROTOCOL",
        "rubric-revision-v2",
        ("exp.optimize.evaluation.variance.rubric_revision",),
        "069f8829e0c78daca62e1aab486cce48cc179619da44166b4885593bde675b86",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rubric_critic:CRITIC_PROTOCOL",
        "rubric-critic-v1",
        ("exp.optimize.evaluation.variance.rubric_critic",),
        "a7eceb7ba19a6ad1d952fc2fb0f06188d2bfda0d3e5b6bb657edae188e86b57e",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.behavior_profiles:BEHAVIOR_LIBRARY_PROTOCOL",
        "behavior-library-v2",
        ("exp.optimize.evaluation.variance.behavior_profiles",),
        "d84d486a42dff4809f1f090a7ce97b000074fdc6806c6a2d20dfd3f486a2a689",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.behavior_labeling:LABELING_PROTOCOL",
        "behavior-labeling-v1",
        ("exp.optimize.evaluation.variance.behavior_labeling",),
        "dd3b7bf5ee6f97f5abd9534ee5c53fcd1b00e9104753334c6f1c7bb9fa0fdc0c",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.behavior_curation:CURATION_PROTOCOL",
        "behavior-curation-v1",
        ("exp.optimize.evaluation.variance.behavior_curation",),
        "c7995d0cc399a4cabd2e4995f1602a032817abc84d193ff846b84755840272f2",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.world_freeze:WORLD_FREEZE_PROTOCOL",
        "world-freeze-v2",
        ("exp.optimize.evaluation.variance.world_freeze",),
        "85bf882a74c23f2191e70c3d2921a29b7eb42bda5e5a30af5dbb895d04b4eabe",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.world_agent:WORLD_AGENT_PROTOCOL",
        "world-agent-v1",
        ("exp.optimize.evaluation.variance.world_agent",),
        "2ba701ac22dc1654602e6ba4feffd0518834ba977907f97accaea63397a027ec",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.challenges:CHALLENGE_PROTOCOL",
        "rule-challenges-v1",
        ("exp.optimize.evaluation.rules.challenges",),
        "f6fd09c20d5584d8009ae756762d3d307d4ade09ce8ff8ad83d4d6a038449e1c",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rewards:SELECTOR_OBJECTIVE",
        "selector-decision-variance-v1",
        (
            "exp.optimize.evaluation.variance.rewards:SelectorObjective",
            "exp.optimize.evaluation.variance.rewards:SelectorScore",
            "exp.optimize.evaluation.variance.rewards:decision_variance",
            "exp.optimize.evaluation.variance.rewards:selector_score",
            "exp.optimize.evaluation.variance.records:selector_records",
        ),
        "69eecc5c147f4e5f0ebcc51284fed66801f779b2d75bda5fe2fa82ef273a2f80",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rewards:JOINT_OBJECTIVE",
        "decision-variance-joint-v1",
        (
            "exp.optimize.evaluation.variance.rewards:DecisionVarianceObjective",
            "exp.optimize.evaluation.variance.rewards:OptionCounts",
            "exp.optimize.evaluation.variance.rewards:StateConsequence",
            "exp.optimize.evaluation.variance.rewards:ChoiceShift",
            "exp.optimize.evaluation.variance.rewards:jensen_shannon",
            "exp.optimize.evaluation.variance.rewards:_distribution",
            "exp.optimize.evaluation.variance.rewards:normalized_entropy",
            "exp.optimize.evaluation.variance.rewards:choice_shift",
            "exp.optimize.evaluation.variance.rewards:success_difference",
            "exp.optimize.evaluation.variance.rewards:state_consequence",
            "exp.optimize.evaluation.variance.decision_measure:StateEvidence",
            "exp.optimize.evaluation.variance.decision_measure:StateVerdict",
            "exp.optimize.evaluation.variance.decision_measure:JointVerdict",
            "exp.optimize.evaluation.variance.decision_measure:_state_verdict",
            "exp.optimize.evaluation.variance.decision_measure:joint_verdict",
        ),
        "e39468e82da4504e974fc2e1e9d0b870ba4e8a0040053bbe7636e75d1c6dce3c",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rewards:SELECTOR_OBJECTIVE_V2",
        "selector-decision-variance-v2",
        (
            "exp.optimize.evaluation.variance.rewards:SelectorObjectiveV2",
            "exp.optimize.evaluation.variance.rewards:PointDecision",
            "exp.optimize.evaluation.variance.rewards:explained_variance",
            "exp.optimize.evaluation.variance.rewards:selector_score_v2",
            "exp.optimize.evaluation.variance.rewards:score_selector",
        ),
        "2038b4adffee6578ee1086207542935e5bcdddbb962cce87ade930d75814d130",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.decision_measure:DECISION_MEASURE_PROTOCOL",
        "decision-measure-v1",
        ("exp.optimize.evaluation.variance.decision_measure",),
        "258d1b77a601a3d37484f637c5df1c2371011fa98b9dc78fb08123550bc1fdea",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.family_stages:ROLE_DRAWS_PROTOCOL",
        "variance-role-draws-v1",
        (
            "exp.optimize.evaluation.variance.family_stages:run_role",
            "exp.optimize.evaluation.variance.family_stages:free_continuations",
            "exp.optimize.evaluation.variance.family_stages:slot_evidence",
            "exp.optimize.evaluation.variance.family_stages:decision_document",
            "exp.optimize.evaluation.variance.family_stages:outcome_counts",
            "exp.optimize.evaluation.variance.family_stages:_clarity_schema",
            "exp.optimize.evaluation.variance.family_stages:check_clarity",
            "exp.optimize.evaluation.variance.family_stages:review_clarity",
        ),
        "4f5dc6d65df9534670205d41d3002819c0c530627bc15e71adcff08fc2ea0c3a",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.variant_search:VARIANT_SEARCH_PROTOCOL",
        "variant-search-v2",
        ("exp.optimize.evaluation.variance.variant_search",),
        "b256fc8fc11f49f7b635ce539fa76645b24f40d48ff018841df755c1471e3f7d",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.point_options:POINT_OPTIONS_PROTOCOL",
        "point-options-v1",
        ("exp.optimize.evaluation.variance.point_options",),
        "6a0765dffc7c4456236d35b8a3e3ac52641dc24bfeae0b1286531e45e7b132ab",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.nudges:NUDGE_PROTOCOL",
        "instruction-nudge-v1",
        ("exp.optimize.evaluation.variance.nudges",),
        "ce4fdfcc62c88009420e3620495f3877dc7ac944df5e445bd133a3fa1e81a46b",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.nudge_ladder:NUDGE_LADDER_PROTOCOL",
        "nudge-ladder-v2",
        ("exp.optimize.evaluation.variance.nudge_ladder",),
        "6e3e266e68273ee102b7711aff18abcb16752bf5a6712cb8133c431308e3648c",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.schedule:SCHEDULE_PROTOCOL",
        "variance-schedule-v2",
        (
            "exp.optimize.evaluation.variance.schedule",
            "exp.optimize.evaluation.variance.schedule_rollouts",
        ),
        "d80c51faffa155a2249785be154cf6a287702a6f0f8f260785af422ee1e62233",
    ),
    Protocol(
        "exp.simulation.mining.traceml:TRACEML_PROTOCOL",
        "traceml-v2",
        ("exp.simulation.mining.traceml",),
        "25eea63defef2d529aa32b332a423a68ba1f630cb1b1b8f0dca5a99a6707f6c8",
    ),
    Protocol(
        "exp.simulation.mining.traceml_moves:TRACEML_MOVES_PROTOCOL",
        "traceml-moves-v1",
        ("exp.simulation.mining.traceml_moves",),
        "4296775ff791ab1b25923f40e44af886abf33d1a1c1758f3f4faf42983262808",
    ),
    Protocol(
        "exp.simulation.mining.error_tags:ERROR_TAGS_PROTOCOL",
        "error-tags-v1",
        ("exp.simulation.mining.error_tags",),
        "b592705beb4cb34fee35664aa3e50aaae4a1da20d622c33c47393c7f46e7ee66",
    ),
    Protocol(
        "exp.simulation.mining.tag_clusters:TAG_CLUSTERS_PROTOCOL",
        "tag-clusters-v1",
        ("exp.simulation.mining.tag_clusters", "exp.simulation.mining.tag_cluster_names"),
        "dd52f6a6299fd91d7a1ce62b1b76daa6f869404149ceb21625a16d722831ac90",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.crosstalk:CROSSTALK_PROTOCOL",
        "crosstalk-probe-v1",
        ("exp.optimize.evaluation.variance.crosstalk",),
        "a1ec0982a884e5651380fd2007490706b2893db884da8bdfe4df3f0aca243503",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.pipeline_spec:PIPELINE_PROTOCOL",
        "decision-variance-pipeline-v3",
        (
            "exp.optimize.evaluation.variance.pipeline_spec:DecisionVarianceSpec.slice",
            "exp.optimize.evaluation.variance.pipeline_spec:stage_identity",
            "exp.optimize.evaluation.variance.pipeline:_identities",
            "exp.optimize.evaluation.variance.pipeline:_register",
            "exp.optimize.evaluation.variance.pipeline_admission",
            "exp.optimize.evaluation.variance.pipeline_downstream",
            "exp.optimize.evaluation.variance.pipeline:_execute",
            "exp.optimize.evaluation.variance.pipeline:run_decision_variance",
        ),
        "13029ea86f0a6836c5ffaec312a2edd3cd9244aaca4c2d96740ff9e9ecd395c1",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.system_one:SYSTEM_ONE_PROTOCOL",
        "jev-system-one-v2",
        ("exp.optimize.evaluation.rules.system_one", "exp.simulation.rules.memory_set:Passage"),
        "6a5567fde75643b9dc3cc6533d84b44a9497451d561b7166d3b28b706c0661f1",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.system_two:SYSTEM_TWO_PROTOCOL",
        "jev-system-two-v5",
        (
            "exp.optimize.evaluation.rules.system_two",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.optimize.evaluation.rules.memory_edits:RuleDraft",
            "exp.optimize.evaluation.rules.evidence_pages",
            "exp.optimize.evaluation.factory.authoring:author_request_bytes",
            "exp.optimize.evaluation.factory.authoring:_effective_request",
        ),
        "89e88016ef9095a0391735a236487ce92f9f5e8391a1f4fa6f88dd411e558a13",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.inspectors:INSPECTOR_PROTOCOL",
        "rule-memory-inspectors-v5",
        (
            "exp.optimize.evaluation.rules.inspectors",
            "exp.optimize.evaluation.rules.memory_edits",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.optimize.evaluation.rules.evidence_pages",
            "exp.optimize.evaluation.factory.authoring:author_request_bytes",
            "exp.optimize.evaluation.factory.authoring:_effective_request",
        ),
        "11bf0aee4d68a269a2caa97ec22e3ff874274e173960bc75ebf10158a24bbd56",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.memory_stage:MEMORY_STAGE_PROTOCOL",
        "jev-memory-stage-v3",
        ("exp.optimize.evaluation.rules.memory_stage", "exp.optimize.evaluation.rules.governance"),
        "ff7ca2bfa4b4a3cfac5ef6f1e699be3b8129b6fe9d92a924a13de7c84cd2439b",
    ),
    Protocol(
        "exp.simulation.rules.recall:RECALL_PROTOCOL",
        "rule-memory-recall-v1",
        ("exp.simulation.rules.recall",),
        "6d0a690a53015d7c7edd1bf498f70a623344fcc214881ac9051329c463d74ef7",
    ),
    Protocol(
        "exp.simulation.rules.memory_set:MEMORY_SET_FORMAT",
        "rule-memory-set-v1",
        ("exp.simulation.rules.memory_set",),
        "60a13ef8b4d2e6724e4cbb8a679fcc93025d08bf0009192bf6fe812b6ae857fd",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.evidence_pages:EVIDENCE_DELIVERY_PROTOCOL",
        "evidence-pages-v2",
        ("exp.optimize.evaluation.rules.evidence_pages",),
        "33c5dd8c38c80be6def3d5619708850217d2935e65bb601ba1082875e4310dc2",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.locator_inventory:LOCATOR_PROTOCOL",
        "locator-prefix-v1",
        ("exp.optimize.evaluation.rules.locator_inventory",),
        "0cb3fedfd83f48d75c7c68ef1d7d496b9433336612817a0b40a26651be2fa1fe",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.bootstrap_prompts:BOOTSTRAP_PROTOCOL",
        "start-bootstrap-v7",
        (
            "exp.optimize.evaluation.variance.bootstrap_prompts",
            "exp.optimize.evaluation.variance.bootstrap_agents",
            "exp.optimize.evaluation.rules.locator_inventory",
            "exp.optimize.evaluation.variance.bootstrap_reads",
        ),
        "5d1813309cb0ae0c4df191e82327792699e0295db229ce4024479e8ea1bbaaeb",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.bootstrap_reads:BOOTSTRAP_READ_PROTOCOL",
        "bootstrap-step-reads-v1",
        ("exp.optimize.evaluation.variance.bootstrap_reads",),
        "a43f000bfcf5592470e3f5919229db9d15d7ff640109594f81bfd3219dc4fbac",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.bootstrap_protocols:BOOTSTRAP_SOURCE_PROTOCOL",
        "bootstrap-source-revision-v10",
        (
            "exp.optimize.evaluation.variance.bootstrap_protocols",
            "exp.optimize.evaluation.variance.bootstrap_resume",
            "exp.optimize.evaluation.variance.bootstrap_plan",
            "exp.optimize.evaluation.variance.bootstrap_admission",
            "exp.optimize.evaluation.variance.bootstrap_selection",
            "exp.optimize.evaluation.variance.pipeline_inputs",
        ),
        "eb1b43e4a18222ebe4ff0cd3390ebc9f94d1afb0bcc64fcbbea664044fbe8f01",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.bootstrap_admission:BOOTSTRAP_ADMISSION_PROTOCOL",
        "bootstrap-cost-admission-v1",
        (
            "exp.optimize.evaluation.variance.bootstrap_admission",
            "exp.optimize.evaluation.variance.bootstrap_plan:admit_bootstrap_preparation",
            "exp.optimize.evaluation.variance.bootstrap_plan:run_bootstrap_preparation",
        ),
        "d94589a4998e702cd77f75754e2a99029cdc48092a41d33a60f33aef75ec7f66",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.direct_research:DIRECT_RESEARCH_PROTOCOL",
        "full-trajectory-research-v1",
        ("exp.optimize.evaluation.rules.direct_research",),
        "0d3ad4285470877e106137ed46e90f789ac73d5a326d9f7d147cf44d02536920",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.rule_tags:RULE_TAG_PROTOCOL",
        "discovered-rule-tags-v2",
        ("exp.optimize.evaluation.rules.rule_tags",),
        "4833ff6460f427ea989b581ff4f4bf2a3dac75894e81abbed7eb88a6826323f9",
    ),
)


def _resolve(reference: str) -> Source:
    """Import ``module`` or ``module:qualname``."""
    module_name, _, qualname = reference.partition(":")
    target: Source = importlib.import_module(module_name)
    for part in qualname.split(".") if qualname else ():
        target = getattr(target, part)
    return target


def covered_digest(protocol: Protocol) -> str:
    """SHA-256 over each covered source, in order, with its reference as a separator."""
    digest = hashlib.sha256()
    for reference in protocol.covers:
        source = inspect.getsource(_resolve(reference))
        digest.update(reference.encode() + b"\0" + source.encode() + b"\0")
    return digest.hexdigest()


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda p: p.constant.rpartition(":")[2])
def test_protocol_covers_unchanged_source_or_was_bumped(protocol: Protocol) -> None:
    """A covered source change needs a bumped protocol, or a recorded digest for a refactor."""
    value = _resolve(protocol.constant)
    assert value == protocol.value, (
        f"{protocol.constant} is {value!r} but the table records {protocol.value!r}; after a bump, "
        f"record the new value and digest {covered_digest(protocol)}"
    )
    actual = covered_digest(protocol)
    assert actual == protocol.digest, (
        f"source covered by {protocol.constant} changed (digest {actual}). If behavior changed, "
        "bump the constant so saved results re-key, then record the new value and this digest. "
        "If it is a pure refactor (move, imports, formatting, comments), record this digest only."
    )


def test_a_covered_source_change_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing covered source text changes the digest the guard compares."""
    protocol = PROTOCOLS[2]
    before = covered_digest(protocol)
    original = inspect.getsource

    def edited(target: Source) -> str:
        """Return the covered source with one extra statement."""
        return original(target) + "\n    pass\n"

    monkeypatch.setattr(inspect, "getsource", edited)
    assert covered_digest(protocol) != before

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
        "exp.optimize.evaluation.variance.behavior_differences:BEHAVIOR_DIFFERENCES_PROTOCOL",
        "behavior-differences-v1",
        (
            "exp.optimize.evaluation.variance.behavior_differences",
            "exp.optimize.evaluation.variance.behavior_pairs",
            "exp.optimize.evaluation.variance.pipeline_behaviors",
        ),
        "cb3a77644e7d4f623d0c4220ed04e55d156e4932154a29d9ee88a6ab53c79bd5",
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
        "variance-roles-v7",
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
        "cd0fddb7297c7cde2a651402b0c7018a727cdc12b85faad7ea5ad76d34950075",
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
        "rules-agent-loop-v2",
        ("exp.optimize.evaluation.rules.agent_loop",),
        "a08740ad13cbf4105402aa06dd729c8f92801ac9a1711ddb39691cac1a37163e",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.research:RESEARCH_PROTOCOL",
        "rule-research-v2",
        (
            "exp.optimize.evaluation.rules.research",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.simulation.rules.contracts:trajectory_view",
            "exp.simulation.rules.model",
        ),
        "3e2c9b551bf7c5d95dcda3be16ef6334b08925f0c86eaf90ae82c6429268a9c8",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.weighing:WEIGHING_PROTOCOL",
        "rule-weighing-v2",
        (
            "exp.optimize.evaluation.rules.weighing",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.simulation.rules.merge:check_weighing",
            "exp.simulation.rules.model",
        ),
        "35aba72f231764d877f5db88799215a0705c8fe2077bc385e22acf7fc3591e94",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.condense:CONDENSATION_PROTOCOL",
        "rule-condensation-v2",
        (
            "exp.optimize.evaluation.rules.condense",
            "exp.optimize.evaluation.rules.agent_loop",
        ),
        "b93a2595d625a3684b90af702865b4954696bc8dc8a8e6c1d873652353f4f020",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.authors:AUTHORS_PROTOCOL",
        "variance-authors-v8",
        (
            "exp.optimize.evaluation.variance.authors",
            "exp.optimize.evaluation.variance.contracts",
            "exp.optimize.evaluation.variance.coalition",
            "exp.simulation.rules.tool_names",
        ),
        "a985d23d51a987dc308feb8b5965196c39694eaad6c1eb75cfffd815abbbc6ce",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rollout_ingest:ROLLOUT_INGEST_CONTRACT",
        "rollout-ingest-v1",
        ("exp.optimize.evaluation.variance.rollout_ingest",),
        "5d7de906e889047ec69f95e127e8599ca3279379eb67ddaa515ca3ed3e5382c8",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rubric_versions:RUBRIC_VERSION_PROTOCOL",
        "rubric-version-v1",
        ("exp.optimize.evaluation.variance.rubric_versions",),
        "ec562621199d804e4954eefd622767e1aa47c26e31538bcc6cf1d27de75c2e61",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.rubric_evolution:RUBRIC_EVOLUTION_PROTOCOL",
        "rubric-evolution-v1",
        (
            "exp.optimize.evaluation.variance.rubric_evolution",
            "exp.optimize.evaluation.variance.rubric_boundary",
            "exp.optimize.evaluation.variance.rubric_evo_states",
            "exp.optimize.evaluation.variance.rubric_audit",
            "exp.optimize.evaluation.variance.rubric_signatures",
            "exp.optimize.evaluation.variance.rubric_items",
            "exp.optimize.evaluation.variance.rubric_grading",
        ),
        "bb443860b090f62905099d2d1d3853971db5871b8036b0771b411d1af51290dd",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.tag_monitor_contracts:TAG_MONITOR_PROTOCOL",
        "tag-monitor-v1",
        (
            "exp.optimize.evaluation.variance.tag_monitor",
            "exp.optimize.evaluation.variance.tag_monitor_contracts",
            "exp.optimize.evaluation.variance.tag_monitor_stats",
            "exp.optimize.evaluation.variance.tag_monitor_reward_stats",
            "exp.optimize.evaluation.variance.tag_monitor_review",
            "exp.optimize.evaluation.variance.tag_monitor_states",
            "exp.optimize.evaluation.variance.tag_monitor_epoch",
        ),
        "cee84b2903dd594855ce003eb0d15aa1b4b81f63610e6a4d8698ea282e159c9e",
    ),
    Protocol(
        "exp.simulation.mining.tag_health:TAG_HEALTH_PROTOCOL",
        "tag-health-v1",
        ("exp.simulation.mining.tag_health",),
        "1c82c7b87689f4168bad82bb8094d2112053f7f0a4a8789ae24734d86e21ad4d",
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
        "b151845a4b71fbb0e4178e1031af8278e2f8796a6c17f82aef449d490796207d",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.astra_prompt_contracts:ASTRA_PROMPT_PROTOCOL",
        "astra-prompt-refinement-v2",
        (
            "exp.optimize.evaluation.variance.astra_prompt_contracts",
            "exp.optimize.evaluation.variance.astra_prompt_states",
            "exp.optimize.evaluation.variance.astra_prompt_attempts",
            "exp.optimize.evaluation.variance.astra_prompt_draws",
            "exp.optimize.evaluation.variance.astra_prompt_leakage",
            "exp.optimize.evaluation.variance.astra_prompt_calibration",
            "exp.optimize.evaluation.variance.astra_prompt_loop",
            "exp.optimize.evaluation.variance.refined_prompts",
        ),
        "c1ffacb6845f4da3428c584a806e4e6f0769fe0abf66fe47f495a21b1bd885fb",
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
        "b94deb5217263bb506e856df12b93d718ad1b312629475fe82f00471f161e464",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.crosstalk:CROSSTALK_PROTOCOL",
        "crosstalk-probe-v1",
        ("exp.optimize.evaluation.variance.crosstalk",),
        "a1ec0982a884e5651380fd2007490706b2893db884da8bdfe4df3f0aca243503",
    ),
    Protocol(
        "exp.optimize.evaluation.variance.pipeline_spec:PIPELINE_PROTOCOL",
        "decision-variance-pipeline-v2",
        (
            "exp.optimize.evaluation.variance.pipeline_spec:DecisionVarianceSpec.slice",
            "exp.optimize.evaluation.variance.pipeline_spec:stage_identity",
            "exp.optimize.evaluation.variance.pipeline:_identities",
            "exp.optimize.evaluation.variance.pipeline:_register",
        ),
        "96e3c2d58004e42a6af54858515f7be6ad8f8cd53057237515fcc0db138748e8",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.system_one:SYSTEM_ONE_PROTOCOL",
        "jev-system-one-v1",
        (
            "exp.optimize.evaluation.rules.system_one",
            "exp.simulation.rules.memory_set:Passage",
        ),
        "be0bfcb9d90af6b4f9e9db253ca9967d7c78d0d3dba815504dff16c9b8f972dd",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.system_two:SYSTEM_TWO_PROTOCOL",
        "jev-system-two-v2",
        (
            "exp.optimize.evaluation.rules.system_two",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.optimize.evaluation.rules.memory_edits:RuleDraft",
        ),
        "ec14e5771bd4d6209959542a417c1a4dae16eb8f695814f776ca1370253a16dc",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.inspectors:INSPECTOR_PROTOCOL",
        "rule-memory-inspectors-v2",
        (
            "exp.optimize.evaluation.rules.inspectors",
            "exp.optimize.evaluation.rules.memory_edits",
            "exp.optimize.evaluation.rules.agent_loop",
        ),
        "c085c711fb83220740e304f97403102ee04bbf5b6229bf311f6a1f76cc80d773",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.memory_stage:MEMORY_STAGE_PROTOCOL",
        "jev-memory-stage-v3",
        (
            "exp.optimize.evaluation.rules.memory_stage",
            "exp.optimize.evaluation.rules.governance",
            "exp.optimize.evaluation.rules.memory_admission",
        ),
        "932fcfcf599f8a90e28cfc333e4860ef06d79bd2e317b7c1c145110acf46ca80",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.rule_tags:RULE_TAGS_PROTOCOL",
        "rule-tags-v1",
        (
            "exp.optimize.evaluation.rules.rule_tags",
            "exp.optimize.evaluation.rules.agent_loop",
            "exp.simulation.rules.rule_lifecycle",
        ),
        "12aa595aa08d3812622114870437cfce7f543ca5186098453043932bf1367bf0",
    ),
    Protocol(
        "exp.optimize.evaluation.rules.rule_consolidation:CONSOLIDATION_PROTOCOL",
        "rule-consolidation-admitted-v1",
        (
            "exp.optimize.evaluation.rules.rule_consolidation",
            "exp.optimize.evaluation.rules.memory_edits",
        ),
        "bc1cb0828436c4a94b9714d807151907137439d11cb8715548138ec7de4a5a39",
    ),
    Protocol(
        "exp.simulation.rules.admission:ADMISSION_PROTOCOL",
        "rule-tag-admission-v1",
        ("exp.simulation.rules.admission",),
        "6eb1cb832fbe277d664e4a2dc47d5a3d99740a875c5b51543e6c233b29473a36",
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

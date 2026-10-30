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
        "fe55f4aca8ba262638d1c8f3d5c3e56a85529185877fb21483ae095655efc94e",
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
        "89ba387276193e7a3435a0e5617821c0b37db5ad2654aff1509f2a00051f0a74",
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

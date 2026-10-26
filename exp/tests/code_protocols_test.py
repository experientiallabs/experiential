"""Protocol versions stand in for source text in saved-result identities; this guards them.

Calibration, the actor council, task refinement, private workbook views and the restorable
environment backends bind their saved results to a protocol constant instead of hashing their own
source. Saved results then survive moves, formatting and unrelated edits, and re-key only when the
protocol is bumped. Provider attempt timeouts follow the same rule: each admitted physical attempt
records the protocol of the derivation that bounded it.

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
        "exp.optimize.evaluation.tasks.calibration:CALIBRATION_PROTOCOL",
        "task-calibration-v7",
        (
            "exp.optimize.evaluation.tasks.calibration",
            "exp.optimize.evaluation.tasks.admission",
            "exp.simulation.evaluation.calibration",
            "exp.simulation.evaluation.tasks",
            "exp.simulation.evaluation.admission",
        ),
        "04713c2d37d0f5512507a26caa1ba2ea63cd18ec26eeb806f9df923773317328",
    ),
    Protocol(
        "exp.optimize.evaluation.tasks.challenges:CHALLENGE_PROTOCOL",
        "judge-challenge-v6",
        (
            "exp.optimize.evaluation.tasks.challenges",
            "exp.simulation.evaluation.calibration:task_decision",
            "exp.simulation.evaluation.calibration:blinded_case",
        ),
        "cfb50e4ac7dea292a349fd1b25dd4800be6206be76e07a6672d08ef071a041bf",
    ),
    Protocol(
        "exp.optimize.evaluation.council.runner:COUNCIL_HARNESS_PROTOCOL",
        "council-harness-v4",
        (
            "exp.runtime.agents.chat:ChatAgentRuntime",
            "exp.runtime.agents.pi:PiAgentRuntime",
            "exp.simulation.actor:_RecordingEnvironment",
            "exp.simulation.actor:run_actor_workflow",
            "exp.simulation.world_model.grounded_session:open_grounded_session",
        ),
        "c3cc35dd94e6cc7583b095211d7370ffc29672575d3cf39d2beb0f34977f5259",
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
        "sample-count-v1",
        ("exp.optimize.evaluation.learning.sample_count",),
        "0d6a436e42eb2fafe530930865ad937351745c78d81aa83495007a1c753daef8",
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
        "exp.runtime.environments.record_world:RECORD_WORLD_PROTOCOL",
        "sqlite-record-world-v1",
        ("exp.runtime.environments.record_world",),
        "8fb247418212f90d8963088e1323a0b4581dbaf937fc8cda590d784a31867b99",
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

"""Protocol versions stand in for source text in saved-result identities; this guards them.

Calibration, the actor council, task refinement and the restorable environment backends bind
their saved results to a protocol constant instead of hashing their own source. Saved results then
survive moves, formatting and unrelated edits, and re-key only when the protocol is bumped.
Provider attempt timeouts follow the same rule: each admitted physical attempt records the
protocol of the derivation that bounded it.

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
        "task-calibration-v3",
        (
            "exp.optimize.evaluation.tasks.calibration",
            "exp.optimize.evaluation.tasks.admission",
            "exp.simulation.evaluation.calibration",
            "exp.simulation.evaluation.tasks",
            "exp.simulation.evaluation.admission",
        ),
        "b01624522a004f1f4bdb72aefb4f91eb76902b5239fb6ef392a3bc90804853e4",
    ),
    Protocol(
        "exp.optimize.evaluation.tasks.challenges:CHALLENGE_PROTOCOL",
        "judge-challenge-v1",
        (
            "exp.optimize.evaluation.tasks.challenges",
            "exp.simulation.evaluation.calibration:task_decision",
            "exp.simulation.evaluation.calibration:blinded_case",
        ),
        "183baa5acc0f87a6a9ddc5d650e47712228e1d84479c0dcf0ccc151431f74a02",
    ),
    Protocol(
        "exp.optimize.evaluation.council.runner:COUNCIL_HARNESS_PROTOCOL",
        "council-harness-v3",
        (
            "exp.runtime.agents.chat:ChatAgentRuntime",
            "exp.runtime.agents.pi:PiAgentRuntime",
            "exp.simulation.actor:run_actor_workflow",
            "exp.simulation.world_model.grounded_session:open_grounded_session",
        ),
        "b048a8d7ba2aaf0b5136c71684c53983a2d8cc532a9e17d504f8dccaaeb2da8e",
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
        "cb2ad6e8c7435817a17943b6a440dc68ee15aa83cf73b02513c1707c296c43e9",
    ),
    Protocol(
        "exp.optimize.evaluation.learning.sample_count:SAMPLE_COUNT_PROTOCOL",
        "sample-count-v1",
        ("exp.optimize.evaluation.learning.sample_count",),
        "0d6a436e42eb2fafe530930865ad937351745c78d81aa83495007a1c753daef8",
    ),
    Protocol(
        "exp.runtime.models.providers.attempt_timeouts:ATTEMPT_TIMEOUT_PROTOCOL",
        "provider-attempt-timeout-v2",
        ("exp.runtime.models.providers.attempt_timeouts",),
        "792a67af91f4a19d1935306f560b12af0c3785395709032940de85322e7c68a3",
    ),
    Protocol(
        "exp.runtime.environments.workbook_edits:XLSX_ROW_EDITS_PROTOCOL",
        "xlsx-row-edits-v3",
        ("exp.runtime.environments.workbook_edits",),
        "e98496c391918efdef29f6425793c7c8c41e10b3114ce167065fa0c475127abe",
    ),
    Protocol(
        "exp.runtime.environments.artifact_world:ARTIFACT_WORLD_PROTOCOL",
        "artifact-world-v2",
        ("exp.runtime.environments.artifact_world",),
        "b57a74dbfb1482e69a5f11e2e70a60a0c24bbc2bcf8d1beac229fe0d754b0710",
    ),
    Protocol(
        "exp.runtime.environments.record_world:RECORD_WORLD_PROTOCOL",
        "sqlite-record-world-v1",
        ("exp.runtime.environments.record_world",),
        "7a97535d9a1e939a0b9b1d812e1ff326b06cd89866618df8325e97f417add01f",
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

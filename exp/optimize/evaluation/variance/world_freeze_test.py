"""Offline tests for freezing an authored world and reviewing it against an authored rubric."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.variance.authoring_packet import AuthoringPacket
from exp.optimize.evaluation.variance.authoring_packet_test import ticket_packet, tool_reply, wire
from exp.optimize.evaluation.variance.rubric_author import RUBRIC_TOOL, author_rubric
from exp.optimize.evaluation.variance.rubric_author_test import criterion, rubric_answer
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    GroundedFixture,
    grounded_fixture,
)
from exp.optimize.evaluation.variance.world_author import (
    WORLD_TOOL,
    author_world,
    load_world_draft,
    world_draft_reference,
)
from exp.optimize.evaluation.variance.world_author_test import CARELESS_CLOSE, CLOSE, world_answer
from exp.optimize.evaluation.variance.world_freeze import freeze_world, review_compatibility
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.simulation.induction.sandbox import ProgramSandbox


def _world(
    fixture: GroundedFixture,
    monkeypatch: pytest.MonkeyPatch,
    sandbox: ProgramSandbox,
    *,
    close: str = CLOSE,
    bindings: tuple[str, ...] = ("tickets.tier", "tickets.lock"),
) -> tuple[AuthoringPacket, ArtifactInput, ArtifactInput]:
    """Author a ready ticket world and return the packet, its reference and the world draft."""
    packet, reference = ticket_packet(fixture)
    wire(
        monkeypatch,
        tool_reply(WORLD_TOOL, world_answer(packet, close=close, bindings=bindings)),
    )
    draft = author_world(
        fixture.project,
        fixture.settings,
        reference,
        sandbox=sandbox,
        identity="freeze-world",
        code_revision="test",
    )
    assert draft.status == "ready", draft.failure
    return packet, reference, world_draft_reference(fixture.project, draft)


def test_a_coherent_world_freezes_and_every_variant_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate probes resolve to refusals, and each variant start restores to its exact state."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        packet, reference, world = _world(fixture, monkeypatch, sandbox)
        freeze = freeze_world(
            fixture.project,
            reference,
            world,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
            actor=ChatAgentRuntime(maximum_model_calls=6),
        )
        replay = freeze_world(
            fixture.project,
            reference,
            world,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
            actor=ChatAgentRuntime(maximum_model_calls=6),
        )

    assert replay == freeze
    assert not freeze.problems and freeze.trainable
    assert [(row.variant_id, row.status, row.restored) for row in freeze.variants] == [
        ("urgent", "frozen", True),
        ("routine", "frozen", True),
    ]
    resolutions = {(probe.variant_id, probe.case): probe.resolution for probe in freeze.probes}
    assert resolutions == {
        ("urgent", "variant"): "consistent",
        ("urgent", "missing:/tickets/lock"): "refused",
        ("routine", "variant"): "consistent",
        ("routine", "missing:/tickets/lock"): "refused",
    }
    assert all(result.violated == 0 for result in freeze.properties.results)
    assert {variant.variant_id for variant in packet.variants} == {
        row.variant_id for row in freeze.variants
    }


def test_freeze_rejects_a_world_that_silently_allows_an_unauthorized_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a ticket whose lock state is missing is a silent sensitive write: rejected."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        _, reference, world = _world(fixture, monkeypatch, sandbox, close=CARELESS_CLOSE)
        freeze = freeze_world(
            fixture.project,
            reference,
            world,
            sandbox=sandbox,
            code_revision="test",
            base=fixture.definition,
        )

    assert not freeze.trainable
    assert {row.status for row in freeze.variants} == {"rejected"}
    assert all(row.checkpoint is None for row in freeze.variants)
    allowed = [probe for probe in freeze.probes if probe.resolution == "silently_allowed"]
    assert {probe.case for probe in allowed} == {"missing:/tickets/lock"}
    assert any("tickets.lock" in problem for problem in freeze.problems)


def test_without_a_runtime_the_world_stays_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoration that was never proven cannot qualify a variant for training."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        _, reference, world = _world(fixture, monkeypatch, sandbox)
        freeze = freeze_world(
            fixture.project, reference, world, sandbox=sandbox, code_revision="test"
        )

    assert not freeze.trainable and not freeze.problems
    assert {row.status for row in freeze.variants} == {"diagnostic"}
    assert all(row.restored is None for row in freeze.variants)


def test_compatibility_review_records_disagreements_without_reconciling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A criterion needing an unrecorded receipt and an unbound rule are both kept."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    with ProgramSandbox() as sandbox:
        packet, reference, world_ref = _world(
            fixture, monkeypatch, sandbox, bindings=("tickets.tier",)
        )
    needy = criterion(
        "lock.respected",
        question="Did the agent leave a held ticket open?",
        supported_by=["rule:tickets.lock"],
        evidence_needed=[{"kind": "receipt", "detail": "lock receipt"}],
    )
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, rubric_answer(criterion("handled"), needy)),
        tool_reply(RUBRIC_TOOL, rubric_answer()),
    )
    disagreeing = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-needy", code_revision="t"
    )
    agreeing = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-plain", code_revision="t"
    )
    world = load_world_draft(fixture.project, world_ref).world
    assert world is not None and disagreeing.rubric is not None and agreeing.rubric is not None

    review = review_compatibility(packet, world, disagreeing.rubric)
    clean = review_compatibility(packet, world, agreeing.rubric)

    assert not review.compatible
    assert {(row.kind, row.subject) for row in review.disagreements} == {
        ("unobservable_evidence", "lock.respected"),
        ("unbound_rule", "tickets.lock"),
    }
    assert review.world_sha256 == world.identity
    assert review.rubric_sha256 == disagreeing.rubric.identity
    assert clean.compatible and not clean.disagreements

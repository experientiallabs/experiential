"""Offline tests for agentic behavior labeling over the grounded ticket coalition."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue
from exp.common.project import artifact_input
from exp.optimize.evaluation.variance.authoring_packet_test import (
    plan,
    ticket_coalition,
    ticket_packet,
    tool_reply,
    wire,
)
from exp.optimize.evaluation.variance.behavior_labeling import (
    LABEL_TOOL,
    LABELER_VARIANTS,
    label_behaviors,
    labeling_protocol_sha256,
    load_label_run,
)
from exp.optimize.evaluation.variance.behavior_profiles_test import LINTER
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.rubric_critic_test import sent_user
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture

NEW_LINTER: JsonObject = {
    "label_id": "acts-before-reading",
    "behavior": "The agent changed the ticket before reading the state that decides how.",
    "valence": "bad",
    "scope": "cross_task",
    "target": "new_linter",
    "draft": {
        "linter_id": "act.before-reading",
        "title": "Acts before reading",
        "description": "Changes state before reading the facts that decide the change.",
        "detection": "A successful state-changing call precedes any read of the deciding field.",
        "evidence_needed": [{"kind": "tool_call", "detail": "order of reads and changes"}],
        "applies_when": "A decision depends on state the agent can read first.",
        "required_when": "The task or a rule makes the decision depend on that state.",
    },
    "examples": [
        {
            "locator": "attempt:glm-flash.g0",
            "events": ["step:1"],
            "polarity": "positive",
            "note": "Closed the ticket without reading its tier.",
        },
        {
            "locator": "pattern:read-then-escalate",
            "polarity": "negative",
            "note": "Read the tier first, then escalated.",
        },
    ],
    "rationale": "The failing attempt's mistake is exactly the missing read; passes read first.",
}


def _label(**update: JsonValue) -> JsonObject:
    """The new-linter label with fields replaced."""
    return {**NEW_LINTER, **update}


def test_labeling_proposes_a_linter_from_mistakes_and_repairs_a_refused_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each variant labels independently; an invented locator gets one repair draw."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture, coalition=ticket_coalition())
    invented = _label(
        examples=[{"locator": "attempt:invented", "polarity": "positive", "note": "Made up."}]
    )
    transport = wire(
        monkeypatch,
        tool_reply(LABEL_TOOL, {"labels": [NEW_LINTER], "notes": "One behavior."}),
        tool_reply(LABEL_TOOL, {"labels": [invented], "notes": "One behavior."}),
        tool_reply(LABEL_TOOL, {"labels": [NEW_LINTER], "notes": "Repaired."}),
    )
    variants = ("mistakes-first", "user-friction-first")

    run = label_behaviors(
        fixture.project, fixture.settings, reference, exposure=plan(), library=None,
        identity="labels", code_revision="t", variants=variants,
    )  # fmt: skip
    replay = label_behaviors(
        fixture.project, fixture.settings, reference, exposure=plan(), library=None,
        identity="labels", code_revision="t", variants=variants,
    )  # fmt: skip

    assert replay == run and len(transport.requests) == 3
    assert [row.evidence_id for row in run.labels] == [
        "ticket-tier/mistakes-first/acts-before-reading",
        "ticket-tier/user-friction-first/acts-before-reading",
    ]
    first = run.labels[0]
    assert first.kind == "label" and first.family_id == "ticket-tier"
    assert [(e.locator, e.polarity) for e in first.examples] == [
        ("attempt:glm-flash.g0", "positive"),
        ("pattern:read-then-escalate", "negative"),
    ]
    assert first.hint["target"] == "new_linter"
    phases = [(r.variant, r.phase, r.failure is None) for r in run.records]
    assert phases == [
        ("mistakes-first", "label", True),
        ("user-friction-first", "label", False),
        ("user-friction-first", "repair", True),
    ]
    assert "attempt:invented" in str(run.records[1].failure)
    sent = sent_user(transport.requests[0].payload)
    assert "coalition" in sent and "behavior_library" in sent and "glm-flash.g0" in str(sent)
    stored = artifact_input(fixture.project.artifacts.read("labels-result").manifest)
    assert load_label_run(fixture.project, stored) == run
    assert len(LABELER_VARIANTS) == 3 and len(labeling_protocol_sha256()) == 64


def test_labels_naming_actors_or_unknown_linters_are_dropped_and_held_out_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label naming a tier is refused twice and dropped; final-evaluation behavior never runs."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture, coalition=ticket_coalition())
    named = _label(behavior="Like glm-flash, it closed before reading.")
    unknown = _label(target="existing_linter", draft=None, linter_id=LINTER)
    wire(
        monkeypatch,
        tool_reply(LABEL_TOOL, {"labels": [named], "notes": "n"}),
        tool_reply(LABEL_TOOL, {"labels": [unknown], "notes": "n"}),
    )

    run = label_behaviors(
        fixture.project, fixture.settings, reference, exposure=plan(), library=None,
        identity="dropped", code_revision="t", variants=("recoveries-first",),
    )  # fmt: skip

    assert run.labels == ()
    assert "name actors ['glm-flash']" in str(run.records[0].failure)
    assert f"unknown linter {LINTER}" in str(run.records[1].failure)
    held = ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={"tickets": "final_evaluation"},
        traces={},
    )
    with pytest.raises(ValueError, match="come from final evaluation"):
        label_behaviors(
            fixture.project, fixture.settings, reference, exposure=held, library=None,
            identity="held", code_revision="t",
        )  # fmt: skip

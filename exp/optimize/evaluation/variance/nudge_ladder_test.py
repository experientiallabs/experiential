"""Tests for the nudge ladder: reviewed and measured rungs, the un-nudged target and curriculum."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, stable_id
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.contracts import DecisionState, ScenarioSet
from exp.optimize.evaluation.variance.decision_measure import (
    MEASURE_TYPE,
    DecisionMeasure,
    StateEvidence,
    StateMeasure,
    joint_verdict,
)
from exp.optimize.evaluation.variance.decision_measure_test import (
    FORCED,
    OBJECTIVE,
    decision,
    reply,
    role_fixture,
    scenario,
)
from exp.optimize.evaluation.variance.nudge_ladder import (
    LADDER_TYPE,
    NudgeLadder,
    NudgeLadderSpec,
    NudgeRung,
    build_nudge_ladder,
)
from exp.optimize.evaluation.variance.nudges_test import (
    BEFORE,
    INSTRUCTION,
    VIEW,
    clean,
    leaks,
    realism,
)
from exp.optimize.evaluation.variance.rewards import OptionCounts
from exp.optimize.evaluation.variance.schedule import curriculum_tier, nudge_curriculum

STRONG = "The customer mentioned their account was upgraded last month."
"""A strong nudge toward the tier."""

MEDIUM = "Refund the order, the tier allows it."
"""A nudge that states the action; the leak reviewer refuses it."""

WEAK = "The customer said they have been with us a while."
"""A faint nudge that does not move the actor."""


def _proposal() -> list:
    """The author's reply proposing one nudge per strength."""
    nudges = [
        {"strength": strength, "text": text, "source": "user", "rationale": "Points at tier."}
        for strength, text in (("strong", STRONG), ("medium", MEDIUM), ("weak", WEAK))
    ]
    return [reply([("propose_nudges", {"nudges": nudges, "limitations": ["fixture"]})])]


def _state(sid: str, choices: str) -> StateEvidence:
    """One clear state where forcing refund succeeds far more often than decline."""
    names = {"r": "refund", "d": "decline"}
    return StateEvidence(
        state_id=sid,
        correct_option="refund",
        choices=tuple(names[c] for c in choices),
        options=(
            OptionCounts(option_id="refund", successes=18, planned=20),
            OptionCounts(option_id="decline", successes=2, planned=20),
        ),
        clarity_review=True,
    )


CHOICES = {
    "gold": "dddddddddddd",
    "gold-nudge-0": "rrrrrrrrdddd",
    "gold-nudge-2": "ddddddddddrd",
}
"""Free choices per measured state: the un-nudged actor always declines."""


class _Measurer:
    """Saves a synthetic measure of whatever states it is given, recording each call."""

    def __init__(self, project: ProjectStore) -> None:
        """Bind the project and start with no calls."""
        self.project, self.calls = project, []

    def __call__(
        self, laddered: DecisionState, scenarios: ScenarioSet
    ) -> tuple[ArtifactInput, DecisionMeasure]:
        """Save and return the measure of the given states."""
        self.calls.append((laddered, scenarios))
        states = [_state(s.scenario_id, CHOICES[s.scenario_id]) for s in scenarios.scenarios]
        identity = stable_id("fixture-ladder-measure", {"n": len(self.calls)})
        free = ArtifactInput(artifact_id="free-study", sha256="4" * 64)
        measure = DecisionMeasure(
            identity=identity,
            decision=laddered,
            states=tuple(
                StateMeasure(state_id=s.state_id, free=free, choices=(), reviews=(), evidence=s)
                for s in states
            ),
            verdict=joint_verdict(states, ["refund", "decline"], OBJECTIVE),
        )
        manifest = self.project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=MEASURE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
            ),
            files={"measure.json": measure},
        )
        return artifact_input(manifest), measure


def test_the_ladder_admits_shifting_rungs_refuses_the_rest_and_ends_unnudged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strong shifts and is admitted; medium leaks; weak does not shift; the target is last."""
    replies = [
        *_proposal(),
        clean(),
        realism(),
        realism(),
        leaks("Refund the order"),
        realism(),
        realism(),
        clean(),
        realism(),
        realism(),
    ]
    project, settings, transport = role_fixture(tmp_path, monkeypatch, replies)
    spec = NudgeLadderSpec(
        author=settings,
        leak_reviewer=settings,
        realism_reviewer=settings,
        profile=FORCED,
        leak_variants=("answer-giveaway-first",),
        realism_stances=("customer_voice", "skeptical_editor"),
        maximum_rounds=1,
    )
    measurer = _Measurer(project)
    base = scenario("gold", "refund", "gold")

    def build() -> tuple[ArtifactInput, NudgeLadder]:
        """Run or replay the ladder."""
        return build_nudge_ladder(
            project,
            spec,
            decision(),
            base,
            instruction=INSTRUCTION,
            view=VIEW,
            before=BEFORE,
            measure=measurer,
            identity="ladder-a",
            code_revision="test",
        )

    reference, ladder = build()

    strong, medium, weak, target = ladder.rungs
    assert [r.strength for r in ladder.rungs] == ["strong", "medium", "weak", "none"]
    assert [r.position for r in ladder.rungs] == [0, 1, 2, 3]
    assert strong.admitted and strong.shift is not None and strong.shift.holds
    assert strong.minority is not None and strong.minority >= 0.125
    assert not medium.admitted and medium.state is None
    assert any("nudge leak review" in p for p in medium.problems)
    assert not weak.admitted
    assert any("does not shift the choice" in p for p in weak.problems)
    assert target.nudge is None and target.scenario == base and target.admitted
    assert ladder.curriculum == (strong, target)
    laddered, measured = measurer.calls[0]
    assert [s.scenario_id for s in measured.scenarios] == ["gold", "gold-nudge-0", "gold-nudge-2"]
    assert all(s.correct_option == "refund" for s in measured.scenarios)
    assert laddered.instruction_axis is not None
    assert measured.scenarios[1].instruction == f"{INSTRUCTION}\n\n{STRONG}"
    assert "STANCE (attention_first)" in json.dumps(transport.requests[0].payload)
    assert project.artifacts.read(reference.artifact_id).manifest.artifact_type == LADDER_TYPE

    assert build() == (reference, ladder)
    assert len(transport.requests) == len(replies) and len(measurer.calls) == 1
    with pytest.raises(ValueError, match="strong to weak"):
        NudgeLadderSpec(
            author=settings,
            leak_reviewer=settings,
            realism_reviewer=settings,
            strengths=("weak", "strong"),
        )


def _rung(position: int, strength: str, *, admitted: bool = True) -> NudgeRung:
    """A bare rung of the base scenario at one strength."""
    sid = "gold" if strength == "none" else f"gold-nudge-{position}"
    return NudgeRung.model_validate(
        {
            "position": position,
            "strength": strength,
            "scenario": scenario(sid, "refund", "gold").model_dump(mode="json"),
            "nudge": None
            if strength == "none"
            else {"strength": strength, "text": sid, "source": "user", "rationale": "r"},
            "admitted": admitted,
        }
    )


def test_the_curriculum_fades_from_strong_to_weak_and_ends_on_the_unnudged_targets() -> None:
    """Rungs of every ladder are ordered by strength; tiers advance and stay on the target."""
    first = NudgeLadder(
        identity="first",
        base="gold",
        rounds=(),
        rungs=(_rung(0, "strong"), _rung(1, "weak"), _rung(2, "none")),
    )
    second = NudgeLadder(
        identity="second",
        base="gold",
        rounds=(),
        rungs=(_rung(0, "strong", admitted=False), _rung(1, "medium"), _rung(2, "none")),
    )

    stages = nudge_curriculum([first, second])

    assert [(s.ladder, s.strength) for s in stages] == [
        ("first", "strong"),
        ("second", "medium"),
        ("first", "weak"),
        ("first", "none"),
        ("second", "none"),
    ]
    assert [s.strength for s in nudge_curriculum([first, second], nudged=False)] == ["none"] * 2
    assert [s.strength for s in curriculum_tier(stages, 0, 2)] == ["strong"]
    assert [s.strength for s in curriculum_tier(stages, 2, 2)] == ["medium"]
    assert {s.strength for s in curriculum_tier(stages, 99, 2)} == {"none"}
    assert not any(s.nudged for s in curriculum_tier(stages, 99, 2))
    with pytest.raises(ValueError, match="ends with the un-nudged decision"):
        NudgeLadder(identity="x", base="gold", rounds=(), rungs=(_rung(0, "strong"),))

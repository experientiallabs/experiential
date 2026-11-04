"""Tests for the measured variant search: deficits fed back, bounded rounds, joint acceptance."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, stable_id
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.contracts import (
    DecisionState,
    Scenario,
    ScenarioLimits,
    ScenarioSet,
)
from exp.optimize.evaluation.variance.decision_measure import (
    MEASURE_TYPE,
    DecisionMeasure,
    StateEvidence,
    StateMeasure,
    joint_verdict,
)
from exp.optimize.evaluation.variance.decision_measure_test import (
    OBJECTIVE,
    _evidence,
    decision,
    reply,
    role_fixture,
    scenario,
    spec,
)
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.variant_search import (
    SEARCH_TYPE,
    VariantSearchSpec,
    ladder_measurer,
    measure_feedback,
    search_variants,
)
from exp.runtime.models.providers.transport import JsonHttpResponse

BEFORE: JsonObject = {"account": {"tier": "silver"}}
"""World state just before the decision."""

LIMITS = ScenarioLimits(tools=frozenset({"lookup"}), remaining_calls=4)
"""The actor's tools and remaining calls."""


def _set(*, flip: bool = True) -> JsonObject:
    """A scenario set; without a flip every variant makes refund correct."""
    basic = scenario("basic", "decline" if flip else "refund", "basic")
    return {
        "limitations": ["the simulator's tier semantics are assumed"],
        "scenarios": [
            scenario("gold", "refund", "gold").model_dump(mode="json"),
            basic.model_dump(mode="json"),
        ],
    }


def _author(*, flip: bool = True) -> JsonHttpResponse:
    """A forced author reply submitting a scenario set."""
    return reply([("author_scenarios", _set(flip=flip))])


def _saved(project: ProjectStore, name: str, states: list[StateEvidence]) -> DecisionMeasure:
    """Save a decision measure over synthetic state evidence and return it."""
    verdict = joint_verdict(states, ["refund", "decline"], OBJECTIVE)
    free = ArtifactInput(artifact_id="free-study", sha256="4" * 64)
    identity = stable_id("fixture-measure", {"name": name})
    measure = DecisionMeasure(
        identity=identity,
        decision=decision(),
        states=tuple(
            StateMeasure(state_id=s.state_id, free=free, choices=(), reviews=(), evidence=s)
            for s in states
        ),
        verdict=verdict,
    )
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=MEASURE_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"measure.json": measure},
    )
    return measure


class _Measurer:
    """Scripted measurements, one per call, recording the sets it measured."""

    def __init__(self, project: ProjectStore, rounds: list[list[StateEvidence]]) -> None:
        """Bind the project and the evidence each successive call measures."""
        self.project, self.rounds, self.seen = project, rounds, []

    def __call__(
        self, scenarios: ScenarioSet, number: int
    ) -> tuple[ArtifactInput, DecisionMeasure]:
        """Save and return the next scripted measure."""
        self.seen.append((number, [s.scenario_id for s in scenarios.scenarios]))
        measure = self._next(number)
        return artifact_input(self.project.artifacts.read(measure.identity).manifest), measure

    def _next(self, number: int) -> DecisionMeasure:
        """Return the measure for one round."""
        return _saved(self.project, f"round-{number}", self.rounds[len(self.seen) - 1])


CONFUSING = [
    _evidence("gold", "refund", "rdrdrdrd", 10, 10),
    _evidence("basic", "decline", "drdrdrdr", 9, 11),
]
"""Agents split evenly and no forced option is better: high entropy, no decision."""

NO_CONSEQUENCE = [
    _evidence("gold", "refund", "rrrrrrrr", 10, 10),
    _evidence("basic", "decline", "dddddddd", 10, 10),
]
"""Choices track the state but forcing either option succeeds equally often."""

ADMITTED = [
    _evidence("gold", "refund", "rrrrrrrd", 18, 3),
    _evidence("basic", "decline", "dddddddr", 2, 17),
]
"""Choices shift, the choice changes success, and every state is clear."""


def test_a_confusing_round_is_refused_and_its_deficits_drive_an_accepted_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 0 is pure uncertainty; round 1 sees those deficits and is admitted; replay is free."""
    project, settings, transport = role_fixture(tmp_path, monkeypatch, [_author(), _author()])
    spec = VariantSearchSpec(author=settings, profile=EvidenceToolProfile(), maximum_rounds=3)
    measurer = _Measurer(project, [CONFUSING, ADMITTED])

    reference, search = search_variants(
        project,
        spec,
        decision(),
        BEFORE,
        LIMITS,
        measure=measurer,
        identity="search-a",
        code_revision="test",
    )

    first, second = search.rounds
    assert not first.accepted and first.verdict is not None and not first.verdict.admitted
    assert any("pure uncertainty" in deficit for deficit in first.deficits)
    assert second.accepted and search.accepted_round == 1
    assert search.scenarios is not None and search.measure == second.measure
    assert [row.stance for row in search.rounds] == ["contrast_first", "observability_first"]
    revision = json.dumps(transport.requests[1].payload)
    assert "pure uncertainty" in revision and "round:0" in revision
    assert "STANCE (observability_first)" in revision and "Never maximize uncertainty" in revision
    assert measurer.seen == [(0, ["gold", "basic"]), (1, ["gold", "basic"])]
    assert project.artifacts.read(reference.artifact_id).manifest.artifact_type == SEARCH_TYPE

    again = search_variants(
        project,
        spec,
        decision(),
        BEFORE,
        LIMITS,
        measure=measurer,
        identity="search-a",
        code_revision="test",
    )
    assert again == (reference, search)
    assert len(transport.requests) == 2 and len(measurer.seen) == 2


def test_choices_that_change_without_changing_success_are_never_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every round's choices shift but outcomes do not; the bounded search ends unaccepted."""
    project, settings, _ = role_fixture(tmp_path, monkeypatch, [_author(), _author()])
    spec = VariantSearchSpec(author=settings, profile=EvidenceToolProfile(), maximum_rounds=2)
    measurer = _Measurer(project, [NO_CONSEQUENCE, NO_CONSEQUENCE])

    _, search = search_variants(
        project,
        spec,
        decision(),
        BEFORE,
        LIMITS,
        measure=measurer,
        identity="search-b",
        code_revision="test",
    )

    assert len(search.rounds) == 2 and search.accepted_round is None and search.scenarios is None
    for row in search.rounds:
        assert row.verdict is not None and row.verdict.choice_shift
        assert not row.verdict.consequence and not row.accepted
        assert any("does not change success" in deficit for deficit in row.deficits)


def test_structural_problems_are_fed_back_without_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A set without a flip is never measured; its structural problem reaches the next round."""
    project, settings, transport = role_fixture(
        tmp_path, monkeypatch, [_author(flip=False), _author()]
    )
    spec = VariantSearchSpec(author=settings, profile=EvidenceToolProfile(), maximum_rounds=2)
    measurer = _Measurer(project, [ADMITTED])

    _, search = search_variants(
        project,
        spec,
        decision(),
        BEFORE,
        LIMITS,
        measure=measurer,
        identity="search-c",
        code_revision="test",
    )

    first, second = search.rounds
    assert first.problems and first.measure is None and first.deficits[0].startswith("structural")
    assert second.accepted and measurer.seen == [(1, ["gold", "basic"])]
    assert "no decision variance" in json.dumps(transport.requests[1].payload)


def test_feedback_lists_each_state_and_specs_refuse_unknown_stances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Feedback carries per-state counts and deficits; unknown author stances are refused."""
    project, settings, _ = role_fixture(tmp_path, monkeypatch, [])
    feedback = measure_feedback(_saved(project, "feedback", CONFUSING))

    assert feedback["admitted"] is False
    states = feedback["states"]
    assert isinstance(states, list) and len(states) == 2
    assert isinstance(feedback["deficits"], list) and feedback["deficits"]
    with pytest.raises(ValueError, match="distinct members"):
        VariantSearchSpec(author=settings, stances=("guess",))


def test_a_set_the_binder_refuses_is_reported_and_nothing_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A binder sees the whole set first; a refused or partial binding measures nothing."""
    project, settings, transport = role_fixture(tmp_path, monkeypatch, [])
    seen: list[list[str]] = []

    def partial(decision: DecisionState, scenarios: Sequence[Scenario]) -> dict:
        """Bind nothing, recording the set it was asked for."""
        del decision
        seen.append([s.scenario_id for s in scenarios])
        return {}

    measure = ladder_measurer(
        project,
        spec(settings),
        bind=partial,
        base=ArtifactInput(artifact_id="base", sha256="0" * 64),
        boundaries=(),
        actions={},
        facts=None,
        repeats=1,
        code_revision="test",
    )
    chosen = ScenarioSet.model_validate(_set())

    result = measure(decision(), chosen)

    assert isinstance(result, str) and result.startswith("binding: ")
    assert "gold" in result and seen == [["gold", "basic"]]
    assert transport.requests == []

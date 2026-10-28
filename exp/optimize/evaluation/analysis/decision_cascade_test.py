"""The cascade fit selects a threshold by its registered objective on development lineages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.analysis import decision_cascade as owner
from exp.optimize.evaluation.analysis.decision_cascade import (
    CascadeFitSpec,
    CascadeObjective,
    ThresholdRow,
    estimate_threshold,
    preflight_cascade_fit,
    run_cascade_fit,
    saved_cascade_fit,
)
from exp.optimize.evaluation.analysis.decision_points import (
    DecisionPoint,
    DecisionPointSet,
    Trajectory,
)
from exp.optimize.evaluation.analysis.decision_points_test import (
    WORLDS,
    ScriptedProviders,
    catalog_model,
    main_study,
    project_fixture,
    world,
)
from exp.runtime.agents.cascade import OTHER, RESPOND, DeciderDefinition, NextActionDecision
from exp.runtime.agents.cascade_test import snapshot
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.evaluation.tasks import Verdict

DEVELOPMENT = ("d1", "d2", "d3", "d4")


def systemone(project: ProjectStore) -> DeciderDefinition:
    """Return the fixture System One decider."""
    return DeciderDefinition(kind="systemone", alias="jev", model=catalog_model(project, "jev"))


def drafts(project: ProjectStore) -> DeciderDefinition:
    """Return the fixture small-model drafting decider."""
    return DeciderDefinition(
        kind="sampled_drafts", alias="judge", model=catalog_model(project, "judge"), samples=2
    )


def fit_spec(
    project: ProjectStore,
    studies: tuple[ArtifactInput, ...],
    decider: DeciderDefinition,
    *,
    grid: tuple[float, ...] = (0.5, 0.7, 0.92, 0.99),
    margin: float = 0.1,
    lineages: tuple[str, ...] = DEVELOPMENT,
) -> CascadeFitSpec:
    """Register a fit over saved studies with a point-estimate objective."""
    return CascadeFitSpec(
        sources=studies,
        main_model=catalog_model(project, "world"),
        development_lineages=lineages,
        decider=decider,
        grid=grid,
        objective=CascadeObjective(
            success_margin=margin,
            compliance_margin=margin,
            criterion="point_estimate",
            minimum_lineages=len(lineages),
        ),
        catalog_sha256=sha256_json(
            load_model_catalog(project.model_catalog_path).model_dump(mode="json")
        ),
        maximum_physical_attempts=100,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )


def scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, wrong: float = 0.6
) -> tuple[ProjectStore, tuple[ArtifactInput, ...], ScriptedProviders]:
    """Four development lineages; System One errs at the initial state of d2 to d4."""
    project = project_fixture(tmp_path, monkeypatch)
    studies = tuple(main_study(project, kind, lineage) for lineage in DEVELOPMENT)
    transport = ScriptedProviders(
        kind, {lineage: (world(kind).mistake_key, wrong) for lineage in ("d2", "d3", "d4")}
    )
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: transport)
    return project, studies, transport


@pytest.mark.parametrize("kind", WORLDS)
def test_fit_selects_the_largest_non_inferior_delegation_on_both_world_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Wrong decisions at 0.6 make 0.5 inferior; 0.7 keeps the correction on d1 and delegates
    the most. Corrected mistakes are spliced onto the passing branch from the same state."""
    project, studies, transport = scenario(tmp_path, monkeypatch, kind)
    spec = fit_spec(project, studies, systemone(project))
    report = run_cascade_fit(project, spec, code_revision="test")
    states = len(report.points.states())
    assert len(transport.requests) == states == {"workbook": 8, "tickets": 12}[kind]
    assert [row.feasible for row in report.rows] == [False, True, True, True]
    low, selected = report.rows[0], report.rows[1]
    assert low.success["estimate"] == pytest.approx(-0.25)
    assert low.success["unit_differences"] == {"d1": 0.5, "d2": -0.5, "d3": -0.5, "d4": -0.5}
    assert low.affected_trajectories == 4
    assert low.verified_error == {"workbook": 6, "tickets": 10}[kind]
    assert selected.success["estimate"] == pytest.approx(0.125)
    shares = [row.delegated_share for row in report.rows]
    assert shares == sorted(shares, reverse=True) and shares[0] == 1 and shares[-1] == 0
    assert report.status == "selected" and report.selected == 0.7
    assert report.policy is not None and report.policy.threshold == 0.7
    assert report.policy.decider == spec.decider
    assert report.main_success["estimate"] == 0.5
    expected_compliance = {"tickets": -0.25, "workbook": 0.0}[kind]
    assert low.compliance["estimate"] == pytest.approx(expected_compliance)
    # A saved fit replays without the catalog, credentials or another decider request.
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    assert saved_cascade_fit(project, spec, code_revision="test") == report
    assert run_cascade_fit(project, spec, code_revision="test") == report
    assert len(transport.requests) == states


def test_no_threshold_is_selected_when_none_is_non_inferior_or_none_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong answer above every grid value blocks selection; an unreachable grid delegates
    nothing; neither produces a policy, and there is no fallback to a default threshold."""
    project, studies, _ = scenario(tmp_path, monkeypatch, "workbook", wrong=0.99)
    blocked = run_cascade_fit(
        project, fit_spec(project, studies, systemone(project), grid=(0.5, 0.7)), code_revision="t"
    )
    assert blocked.status == "no_feasible_threshold" and blocked.policy is None
    idle = run_cascade_fit(
        project, fit_spec(project, studies, systemone(project), grid=(0.999,)), code_revision="t"
    )
    assert idle.status == "no_delegation" and idle.policy is None
    assert idle.rows[0].feasible and idle.rows[0].delegated == 0


def test_interrupted_shadow_decision_stays_unknown_and_is_never_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the durable intent leaves that state undelegated, not rebought."""
    project, studies, transport = scenario(tmp_path, monkeypatch, "workbook")
    spec = fit_spec(project, studies, systemone(project))
    original = transport.post

    def crash(url: str, **arguments: object) -> JsonHttpResponse:
        """Interrupt the process during the first decision dispatch."""
        del url, arguments
        raise KeyboardInterrupt

    monkeypatch.setattr(transport, "post", crash)
    with pytest.raises(KeyboardInterrupt):
        run_cascade_fit(project, spec, code_revision="test")
    monkeypatch.setattr(transport, "post", original)
    report = run_cascade_fit(project, spec, code_revision="test")
    assert report.predictions[0].failure == "UnfinishedDispatchIntent"
    assert sum(row.decision is None for row in report.predictions) == 1
    assert len(transport.requests) == len(report.predictions) - 1


def test_drafting_decider_avoids_main_calls_and_fits_on_complete_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unanimous correct drafts agree with every passing branch; ties prefer the higher value."""
    project = project_fixture(tmp_path, monkeypatch)
    studies = tuple(main_study(project, "tickets", lineage) for lineage in DEVELOPMENT)
    transport = ScriptedProviders("tickets")
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: transport)
    report = run_cascade_fit(
        project, fit_spec(project, studies, drafts(project), grid=(0.5, 1.0)), code_revision="t"
    )
    assert report.status == "selected" and report.selected == 1.0
    row = report.rows[1]
    assert row.calls_avoided == row.delegated == row.decisions
    assert row.verified_correct > 0 and row.success["estimate"] == 0.5
    assert {route for route, _ in transport.requests} == {"judge-v1"}


def test_registration_refuses_default_grids_and_undeclared_lineages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grid is explicit and increasing, and the declared lineages must match the sources."""
    project = project_fixture(tmp_path, monkeypatch)
    studies = (main_study(project, "tickets", "d1"),)
    spec = fit_spec(project, studies, systemone(project), lineages=("d1",))
    payload: JsonObject = spec.model_dump(mode="json")
    for grid in ([], [0.7, 0.5], [1.5]):
        with pytest.raises(ValidationError):
            CascadeFitSpec.model_validate({**payload, "grid": grid})
    with pytest.raises(ValueError, match="must equal the lineages"):
        preflight_cascade_fit(
            project, spec.model_copy(update={"development_lineages": ("d1", "d2")})
        )
    with pytest.raises(ValueError, match="frozen snapshot"):
        preflight_cascade_fit(
            project,
            spec.model_copy(
                update={"decider": spec.decider.model_copy(update={"model": spec.main_model})}
            ),
        )


STATE = sha256_json({"state": "first"})
OPTIONS = ("close", "open", RESPOND)


def reference(name: str) -> ArtifactInput:
    """Return a fixture artifact reference."""
    return ArtifactInput(artifact_id=name, sha256=sha256_json(name))


LATER = sha256_json({"state": "second"})


def branch(
    checkpoint: str,
    lineage: str,
    index: int,
    choice: str,
    outcome: Verdict,
    state: str = STATE,
) -> tuple[Trajectory, DecisionPoint]:
    """Return one branch of one study that makes ``choice`` at ``state``.

    Calling it again with the same branch and another state adds a later point of that branch.
    """
    study = reference(f"study-{checkpoint}")
    point = DecisionPoint(
        id=f"{checkpoint}-{index}-{state}",
        study=study,
        branch=index,
        call=reference(f"call-{checkpoint}-{index}-{state}"),
        checkpoint=reference(checkpoint),
        lineage=lineage,
        task_id=f"task-{lineage}",
        state_sha256=state,
        candidates=OPTIONS,
        main_choice=choice,
        main_action_sha256=sha256_json(choice),
        outcome=outcome,
        policy=outcome,
    )
    row = Trajectory(study=study, branch=index, lineage=lineage, outcome=outcome, policy=outcome)
    return row, point


CLOSE = NextActionDecision(
    selected="close", probabilities={"close": 1.0, "open": 0.0, RESPOND: 0.0, OTHER: 0.0}
)


def fit_inputs(
    branches: list[tuple[Trajectory, DecisionPoint]],
    lineages: tuple[str, ...],
    never_delegate: tuple[str, ...] = (),
) -> tuple[DecisionPointSet, CascadeFitSpec]:
    """Return the labelled points of ``branches`` and a threshold-0.9 registration over them."""
    model = snapshot("main-v1")
    points = DecisionPointSet(
        main_model=model,
        studies=tuple(dict.fromkeys(row.study for row, _ in branches)),
        trajectories=tuple(dict.fromkeys(row for row, _ in branches)),
        points=tuple(point for _, point in branches),
    )
    decider = DeciderDefinition(
        kind="sampled_drafts", alias="small", model=snapshot("small-v1"), samples=2
    )
    spec = CascadeFitSpec(
        sources=points.studies,
        main_model=model,
        development_lineages=lineages,
        decider=decider,
        never_delegate=never_delegate,
        grid=(0.9,),
        objective=CascadeObjective(
            success_margin=0.1,
            compliance_margin=0.1,
            criterion="point_estimate",
            minimum_lineages=len(lineages),
        ),
        catalog_sha256=sha256_json("catalog"),
        maximum_physical_attempts=10,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    return points, spec


def offline(
    branches: list[tuple[Trajectory, DecisionPoint]],
    lineages: tuple[str, ...],
    answer: NextActionDecision | None = CLOSE,
    later: NextActionDecision | None = CLOSE,
) -> ThresholdRow:
    """Estimate threshold 0.9 with ``answer`` at the first state and ``later`` at the second."""
    points, spec = fit_inputs(branches, lineages)
    return estimate_threshold(points, {STATE: answer, LATER: later}, 0.9, spec, frozenset())


def test_donors_and_labels_never_cross_worlds_that_share_a_visible_state() -> None:
    """A passing close in world X neither labels nor stands in for a close in world Y."""
    row = offline(
        [branch("world-x", "a", 0, "close", "pass"), branch("world-y", "b", 0, "open", "fail")],
        ("a", "b"),
    )
    assert row.delegated == 2 and row.affected_trajectories == 1
    assert row.verified_correct == 1
    assert row.success["unit_differences"] == {"a": 0.0, "b": 0.0}


def test_unknown_donors_and_main_outcomes_stay_bounds_against_the_cascade() -> None:
    """One passing donor among unknown ones does not score as a certain pass."""
    donors = [branch("world-x", "a", index, "close", "unassessable") for index in range(3)]
    passing = branch("world-x", "a", 3, "close", "pass")
    changed = branch("world-x", "a", 4, "open", "fail")
    unknown_main = branch("world-x", "a", 5, "open", "unassessable")
    row = offline([*donors, passing, changed, unknown_main], ("a",))
    # Donors: three unknown (counted failed) and one pass give 0.25 to each changed trajectory.
    # The changed failure gains 0.25; the unknown main outcome, bounded at 1, loses 0.75. The
    # unchanged pass is paired with itself; unchanged unknowns carry no paired evidence.
    assert row.success["estimate"] == pytest.approx((0.25 + 0.25 + 1.0) / 3 - (0.0 + 1.0 + 1.0) / 3)


def test_a_state_without_a_decider_answer_stops_the_cascade_trajectory() -> None:
    """The deployed cascade stops on an unknown decider dispatch, so the fit keeps no main pass."""
    row = offline([branch("world-x", "a", 0, "close", "pass")], ("a",), answer=None)
    assert row.delegated == 0 and row.stopped_trajectories == 1
    assert row.success["unit_differences"] == {"a": -1.0}
    assert not row.feasible


class TwoCallDrafts(ScriptedProviders):
    """Append a duplicate ``open_ticket`` call to every call the small drafter makes."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer as scripted, then extend each drafted tool call with a restricted one."""
        response = super().post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        choices = response.body["choices"]
        assert isinstance(choices, list) and isinstance(choices[0], dict)
        message = choices[0]["message"]
        assert isinstance(message, dict)
        calls = message.get("tool_calls")
        if payload.get("model") == "judge-v1" and isinstance(calls, list):
            duplicate = {"title": "Duplicate", "priority": "low"}
            calls.append(
                {
                    "id": "dup",
                    "type": "function",
                    "function": {"name": "open_ticket", "arguments": json.dumps(duplicate)},
                }
            )
        return response


def test_the_fit_refuses_drafts_the_runtime_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft whose later call is restricted adds no delegation, avoided call or splice."""
    project = project_fixture(tmp_path, monkeypatch)
    studies = tuple(main_study(project, "tickets", lineage) for lineage in DEVELOPMENT)
    transport = TwoCallDrafts("tickets")
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: transport)
    spec = fit_spec(project, studies, drafts(project), grid=(1.0,))
    admitted = run_cascade_fit(project, spec, code_revision="t").rows[0]
    assert admitted.delegated == admitted.decisions and admitted.affected_trajectories > 0
    estimate = admitted.success["estimate"]
    assert isinstance(estimate, float) and estimate < 0
    report = run_cascade_fit(
        project, spec.model_copy(update={"never_delegate": ("open_ticket",)}), code_revision="t"
    )
    row = report.rows[0]
    initial = sum(point.main_choice != RESPOND for point in report.points.points)
    assert initial > 0
    assert row.delegated == row.calls_avoided == row.decisions - initial
    assert row.affected_trajectories == 0 and row.success["estimate"] == 0


def test_points_after_a_stop_earn_no_delegation() -> None:
    """A stopped episode never reaches its later states, so they add no delegated share."""
    row = offline(
        [
            branch("world-x", "a", 0, "close", "pass"),
            branch("world-x", "a", 0, "close", "pass", state=LATER),
        ],
        ("a",),
        answer=None,
    )
    assert (row.delegated, row.calls_avoided, row.agreement) == (0, 0, None)
    assert row.verified_correct == 0 and row.stopped_trajectories == 1
    assert row.success["unit_differences"] == {"a": -1.0}


def test_a_stopped_donor_and_an_unknown_stopped_main_outcome_count_against_the_cascade() -> None:
    """A donor that stops later stands in as a failure; an unknown stopped outcome is bounded."""
    row = offline(
        [
            branch("world-x", "a", 0, "open", "fail"),
            branch("world-x", "a", 1, "close", "pass"),
            branch("world-x", "a", 1, "close", "pass", state=LATER),
            branch("world-y", "b", 0, "close", "unassessable", state=LATER),
        ],
        ("a", "b"),
        later=None,
    )
    assert row.affected_trajectories == 1 and row.stopped_trajectories == 2
    # a: the changed failure takes its stopped donor's 0, the donor itself stops at 0 against a
    # main pass; b: the unknown main outcome is bounded at 1 against the stopped cascade's 0.
    assert row.success["unit_differences"] == {"a": -0.5, "b": -1.0}


def test_a_trajectory_that_left_the_main_path_is_not_stopped_by_a_later_state() -> None:
    """After a delegated disagreement the cascade follows the donor, not the main path."""
    row = offline(
        [
            branch("world-x", "a", 0, "open", "fail"),
            branch("world-x", "a", 0, "open", "fail", state=LATER),
            branch("world-x", "a", 1, "close", "pass"),
        ],
        ("a",),
        later=None,
    )
    assert row.affected_trajectories == 1 and row.stopped_trajectories == 0
    assert row.success["unit_differences"] == {"a": 0.5}

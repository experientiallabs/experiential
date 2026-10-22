"""The cascade fit selects a threshold by its registered objective on development lineages."""

from __future__ import annotations

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
    preflight_cascade_fit,
    run_cascade_fit,
    saved_cascade_fit,
)
from exp.optimize.evaluation.analysis.decision_points_test import (
    WORLDS,
    ScriptedProviders,
    catalog_model,
    main_study,
    project_fixture,
    world,
)
from exp.runtime.agents.cascade import DeciderDefinition
from exp.runtime.models.providers.transport import JsonHttpResponse

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

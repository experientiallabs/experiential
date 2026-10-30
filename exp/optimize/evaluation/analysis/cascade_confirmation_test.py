"""A fitted cascade is confirmed only by episodes on held-out lineages of two world kinds.

Four held-out lineages give wide lineage intervals, so these fixtures register a wide margin;
the decision rule and its blockers are what is under test, not a realistic margin.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis import decision_cascade
from exp.optimize.evaluation.analysis.cascade_confirmation import (
    CascadeConfirmationSpec,
    confirmation_panel,
    run_cascade_confirmation,
)
from exp.optimize.evaluation.analysis.decision_cascade import run_cascade_fit
from exp.optimize.evaluation.analysis.decision_cascade_test import (
    DEVELOPMENT,
    drafts,
    fit_spec,
    systemone,
)
from exp.optimize.evaluation.analysis.decision_points_test import (
    ACTOR_CALLS,
    WORLDS,
    ScriptedProviders,
    capture,
    catalog_model,
    main_study,
    project_fixture,
    world,
)
from exp.optimize.evaluation.continuations import job as job_owner
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.runtime.agents.cascade import DeciderDefinition

HELD_OUT = ("h1", "h2", "h3", "h4")


def held_out_job(project: ProjectStore, tmp_path: Path, kind: str, lineage: str) -> ContinuationJob:
    """Return a main-model-only job from one held-out lineage's initial checkpoint."""
    checkpoint = capture(project.artifacts, kind, lineage)
    runtime = tmp_path / "cascade_runtime.py"
    runtime.write_text(
        "from exp.optimize.evaluation.analysis.decision_points_test import continuation_runtime\n"
    )
    return ContinuationJob(
        study=ContinuationStudySpec(
            checkpoint=checkpoint,
            actor_model=catalog_model(project, "world"),
            verifier=world(kind).verifier,
            probe_sha256=world(kind).probe_sha256,
            repeats=2,
            selector="held-out initial checkpoints; cascade fixture",
        ),
        runtime_file=runtime,
        runtime_sha256=sha256_bytes(runtime.read_bytes()),
        configuration={"world": kind, "checkpoint": checkpoint.model_dump(mode="json")},
        actor_alias="world",
        catalog_sha256=sha256_json(
            load_model_catalog(project.model_catalog_path).model_dump(mode="json")
        ),
        maximum_model_calls=ACTOR_CALLS,
        maximum_physical_attempts=12,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )


def fitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    decider: str = "systemone",
    held_out: dict[str, tuple[str, float]] | None = None,
) -> tuple[ProjectStore, CascadeConfirmationSpec, ScriptedProviders]:
    """Fit on development lineages, then register held-out cases against the saved fit."""
    project = project_fixture(tmp_path, monkeypatch)
    transport = ScriptedProviders(kind, held_out)
    monkeypatch.setattr(decision_cascade, "HttpxJsonTransport", lambda: transport)
    monkeypatch.setattr(job_owner, "HttpxJsonTransport", lambda: transport)
    studies = tuple(main_study(project, kind, lineage) for lineage in DEVELOPMENT)
    definition: DeciderDefinition = (
        systemone(project) if decider == "systemone" else drafts(project)
    )
    spec = fit_spec(project, studies, definition, grid=(0.5, 0.7, 0.92), margin=0.75)
    report = run_cascade_fit(project, spec, code_revision="test")
    assert report.status == "selected"
    reference = artifact_input(project.artifacts.read(spec.identity("test") + "-result").manifest)
    cases = tuple(held_out_job(project, tmp_path, kind, lineage) for lineage in HELD_OUT)
    return project, CascadeConfirmationSpec(fit=reference, cases=cases), transport


@pytest.mark.parametrize("kind", WORLDS)
def test_cascade_episodes_confirm_non_inferiority_on_held_out_lineages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Forced correct decisions fix the main model's mistakes in the closed loop."""
    project, spec, transport = fitted(tmp_path, monkeypatch, kind)
    report = run_cascade_confirmation(project, spec, code_revision="test")
    main, cascade = report.arms["main model only"], report.arms["decision cascade"]
    assert isinstance(main, dict) and isinstance(cascade, dict)
    assert (main["planned"], main["passes"]) == (8, 4)
    assert (cascade["planned"], cascade["passes"]) == (8, 8)
    assert report.success["estimate"] == 0.5 and report.success["independent_units"] == 4
    assert report.confirmed and report.blockers == ()
    delegation = report.delegation
    assert delegation["delegated"] == delegation["decisions"] == 16
    assert delegation["calls_avoided"] == 0 and delegation["forced_not_followed"] == 0
    assert delegation["fit_delegated_share"] == 1.0
    assert report.policy.threshold == 0.7
    calls = len(transport.requests)
    assert run_cascade_confirmation(project, spec, code_revision="test") == report
    assert len(transport.requests) == calls


def test_reached_state_shift_blocks_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decider confidently wrong on held-out states fails the closed loop despite the fit."""
    wrong = {lineage: (world("tickets").mistake_key, 0.95) for lineage in HELD_OUT}
    project, spec, _ = fitted(tmp_path, monkeypatch, "tickets", held_out=wrong)
    report = run_cascade_confirmation(project, spec, code_revision="test")
    assert not report.confirmed
    assert report.success["estimate"] == -0.5
    assert any(blocker.startswith("task success") for blocker in report.blockers)
    assert any(blocker.startswith("policy compliance") for blocker in report.blockers)


def test_drafted_decisions_avoid_main_calls_in_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small-model draft replaces the main call and records the drafter as the decider."""
    project, spec, transport = fitted(tmp_path, monkeypatch, "tickets", decider="drafts")
    report = run_cascade_confirmation(project, spec, code_revision="test")
    assert report.confirmed
    assert report.delegation["calls_avoided"] == report.delegation["delegated"] == 16
    cascade_main_calls = [
        payload
        for route, payload in transport.requests
        if route == "world-v1" and payload.get("tool_choice") not in (None, "auto")
    ]
    assert not cascade_main_calls


def test_confirmation_refuses_fit_lineages_and_prices_both_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A development lineage cannot confirm its own fit; the estimate covers both arms."""
    project, spec, transport = fitted(tmp_path, monkeypatch, "workbook")
    reused = spec.model_copy(
        update={"cases": (held_out_job(project, tmp_path, "workbook", DEVELOPMENT[0]),)}
    )
    with pytest.raises(ValueError, match="was used to fit"):
        confirmation_panel(project, reused)
    requests = len(transport.requests)
    components = evaluation_costs(project, spec)
    assert {row.name for row in components} >= {"world", "jev"}
    assert sum(row.attempts for row in components if row.name == "world") == 2 * 4 * 12
    assert len(transport.requests) == requests

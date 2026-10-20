"""Actor panels keep identical task inputs, complete denominators and explicit unknown bounds."""

from pathlib import Path

import pytest

from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.actor_panel import (
    ActorPanelSpec,
    PanelActor,
    preflight_actor_panel,
    run_actor_panel,
)
from exp.optimize.evaluation.continuation_job_test import job_fixture
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def panel_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ActorPanelSpec, ScriptedJsonTransport]:
    """Compare two configured snapshots on the same real workbook starting state."""
    project, base, transport = job_fixture(tmp_path, monkeypatch)
    base = base.model_copy(
        update={"study": base.study.model_copy(update={"training_specification": None})}
    )
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    spec = ActorPanelSpec(
        cases=(base,),
        actors=(
            PanelActor(alias="world", model=catalog.snapshot("world")[0], label="declared weak"),
            PanelActor(alias="judge", model=catalog.snapshot("judge")[0], label="declared strong"),
        ),
    )
    return project, spec, transport


def test_actor_panel_does_not_turn_missing_strong_outcomes_into_tier_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown strong-model branches prevent claiming the declared ranking is measured."""
    project, spec, transport = panel_fixture(tmp_path, monkeypatch)
    report = run_actor_panel(project, spec, code_revision="test")
    assert len(report.cells) == 2
    first, second = report.cells
    assert first.job.study.checkpoint == second.job.study.checkpoint
    assert first.job.study.verifier == second.job.study.verifier
    assert first.job.maximum_model_calls == second.job.maximum_model_calls
    assert first.job.system_prompt == second.job.system_prompt
    assert first.job.study.actor_model != second.job.study.actor_model
    actors = report.metrics["actors"]
    assert isinstance(actors, list)
    assert isinstance(actors[0], dict) and actors[0]["successes"] == 1
    assert isinstance(actors[1], dict) and actors[1]["unknown"] == 2
    assert actors[1]["planned"] == 2
    comparisons = report.metrics["comparisons"]
    assert isinstance(comparisons, list) and isinstance(comparisons[0], dict)
    assert comparisons[0]["separates_on_fixed_panel"] is False
    assert "not confidence intervals" in str(report.metrics["bound_meaning"])
    calls = len(transport.requests)
    spec.cases[0].runtime_file.unlink()
    assert run_actor_panel(project, spec, code_revision="test") == report
    assert len(transport.requests) == calls


def test_actor_panel_rejects_duplicate_source_groups_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeating one checkpoint cannot masquerade as coverage of multiple independent cases."""
    project, spec, transport = panel_fixture(tmp_path, monkeypatch)
    duplicated = spec.model_copy(update={"cases": (*spec.cases, *spec.cases)})
    with pytest.raises(ValueError, match="disjoint declared source lineages"):
        preflight_actor_panel(project, duplicated)
    assert not transport.requests


def test_actor_panel_requires_observed_difference_not_declared_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A goal-failing workbook edit versus a valid one gives an exact measured fixed-panel gap."""
    import json

    from exp.runtime.environments.tests.world_conformance_test import workbook_case
    from exp.runtime.models.providers.transport import JsonHttpResponse

    project, spec, _ = panel_fixture(tmp_path, monkeypatch)
    base = spec.cases[0]
    base = base.model_copy(update={"study": base.study.model_copy(update={"repeats": 1})})
    spec = spec.model_copy(update={"cases": (base,)})
    replies = []
    for actor, count in (("world-v1", 2), ("judge-v1", 1)):
        call = workbook_case().first
        arguments = {
            **call.arguments,
            "deletions": [{"sheet": "Orders", "start": 2, "count": count}],
        }
        for message, finish in (
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "tool_calls",
            ),
            ({"role": "assistant", "content": "done"}, "stop"),
        ):
            replies.append(
                JsonHttpResponse(
                    status_code=200,
                    body={
                        "id": f"reply-{len(replies)}",
                        "model": actor,
                        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                    },
                )
            )
    transport = ScriptedJsonTransport(tuple(replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuation_job.HttpxJsonTransport", lambda: transport
    )
    report = run_actor_panel(project, spec, code_revision="test")
    comparison = report.metrics["comparisons"]
    assert isinstance(comparison, list) and isinstance(comparison[0], dict)
    assert comparison[0]["difference_bounds"] == [1.0, 1.0]
    assert comparison[0]["separates_on_fixed_panel"] is True
    assert len(transport.requests) == 4

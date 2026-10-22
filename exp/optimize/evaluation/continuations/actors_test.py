"""Actor panels keep identical task inputs, complete denominators and explicit unknown bounds."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from exp.common.core.artifacts import sha256_json
from exp.common.models import ModelMessage, load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.actors import (
    ActorPanelSpec,
    PanelActor,
    instructed_checkpoint,
    panel_jobs,
    preflight_actor_panel,
    run_actor_panel,
)
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.study import study_identity
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import persist_continuation_checkpoint


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


def test_busy_child_can_resume_without_a_terminal_panel_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real child lease conflict buys nothing and remains retryable after release."""
    project, spec, transport = panel_fixture(tmp_path, monkeypatch)
    identity = spec.identity("test")
    cell = panel_jobs(spec, "test")[0]
    child = study_identity(cell.job.study, cell.job.implementation("test"))
    with FileLock(project.artifacts.project_directory / (child + ".lock"), timeout=0):
        with pytest.raises(Timeout):
            run_actor_panel(project, spec, code_revision="test")
        assert not transport.requests
        assert not project.artifacts.exists(identity + "-cell-0")
    report = run_actor_panel(project, spec, code_revision="test")
    assert report.cells[0].result is not None and report.cells[0].failure is None
    assert len(transport.requests) > 0


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
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )
    report = run_actor_panel(project, spec, code_revision="test")
    comparison = report.metrics["comparisons"]
    assert isinstance(comparison, list) and isinstance(comparison[0], dict)
    assert comparison[0]["difference_bounds"] == [1.0, 1.0]
    assert comparison[0]["separates_on_fixed_panel"] is True
    actors = report.metrics["actors"]
    assert isinstance(actors, list)
    assert [row["actor_budget_exhausted"] for row in actors if isinstance(row, dict)] == [0, 0]
    assert len(transport.requests) == 4


def test_every_actor_capacity_is_checked_before_any_panel_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undersized second actor refuses the panel before the first actor spends."""
    project, spec, transport = panel_fixture(tmp_path, monkeypatch)
    catalog = load_model_catalog(project.model_catalog_path)
    second = catalog.models["judge"]
    assert second.capabilities is not None
    models = dict(catalog.models)
    models["judge"] = second.model_copy(
        update={"capabilities": second.capabilities.model_copy(update={"maximum_output_tokens": 1})}
    )
    catalog = catalog.model_copy(update={"models": models})
    write_model_catalog(project.model_catalog_path, catalog)
    registry = RuntimeModelCatalog(catalog, environment={})
    spec = spec.model_copy(
        update={
            "cases": tuple(
                job.model_copy(
                    update={"catalog_sha256": sha256_json(catalog.model_dump(mode="json"))}
                )
                for job in spec.cases
            ),
            "actors": tuple(
                actor.model_copy(update={"model": registry.snapshot(actor.alias)[0]})
                for actor in spec.actors
            ),
        }
    )
    with pytest.raises(ValueError, match="output allowance exceeds"):
        run_actor_panel(project, spec, code_revision="test")
    assert not transport.requests


def test_one_model_under_two_system_prompts_is_a_panel_of_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each prompted cell starts from its re-instructed checkpoint; duplicates still refuse."""
    project, spec, _ = panel_fixture(tmp_path, monkeypatch)
    actor, case = spec.actors[0], spec.cases[0]
    variant = instructed_checkpoint(project, case, "Check twice.")
    assert variant.actor.messages[0] == ModelMessage(role="system", content="Check twice.")
    reference = persist_continuation_checkpoint(
        project.artifacts, variant, created_at=datetime.now(UTC), code_revision="test"
    )
    edited = actor.model_copy(
        update={"label": "edited", "system_prompt": "Check twice.", "checkpoints": (reference,)}
    )
    policies = spec.model_copy(update={"actors": (actor, edited)})
    cells = panel_jobs(policies, "test")
    assert [cell.job.system_prompt for cell in cells] == [case.system_prompt, "Check twice."]
    assert [cell.job.study.checkpoint for cell in cells] == [case.study.checkpoint, reference]
    assert {cell.job.actor_alias for cell in cells} == {actor.alias}
    preflight_actor_panel(project, policies)
    assert "system_prompt" not in actor.model_dump(mode="json")
    with pytest.raises(ValueError, match="distinct aliases and frozen models"):
        ActorPanelSpec(cases=spec.cases, actors=(actor, actor.model_copy(update={"label": "x"})))
    with pytest.raises(ValueError, match="one re-instructed checkpoint per case"):
        ActorPanelSpec(
            cases=spec.cases, actors=(actor, edited.model_copy(update={"checkpoints": None}))
        )
    stale = edited.model_copy(update={"checkpoints": (case.study.checkpoint,)})
    with pytest.raises(ValueError, match="initial checkpoint under that system prompt"):
        preflight_actor_panel(project, spec.model_copy(update={"actors": (actor, stale)}))


def test_one_model_may_appear_with_and_without_a_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cascade is part of the actor binding; an exact duplicate binding is still refused."""
    from exp.runtime.agents.cascade import CascadePolicy, DeciderDefinition

    project, spec, _ = panel_fixture(tmp_path, monkeypatch)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    plain = spec.actors[0]
    cascade = plain.model_copy(
        update={
            "label": "with cascade",
            "cascade": CascadePolicy(
                decider=DeciderDefinition(
                    kind="systemone", alias="jev", model=catalog.snapshot("jev")[0]
                ),
                threshold=0.8,
            ),
        }
    )
    paired = ActorPanelSpec(cases=spec.cases, actors=(plain, cascade))
    cells = panel_jobs(paired, "test")
    assert [cell.job.study.cascade for cell in cells] == [None, cascade.cascade]
    assert "cascade" not in plain.model_dump(mode="json")
    with pytest.raises(ValueError, match="distinct aliases"):
        ActorPanelSpec(cases=spec.cases, actors=(plain, plain))

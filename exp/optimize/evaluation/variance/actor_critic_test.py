"""End-to-end tests for judged estimation points, critic values and continuations, offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.factory.candidates_test import _reply
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec, run_actor_critic
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def _edit(row: int) -> JsonHttpResponse:
    """One scripted actor reply that deletes a workbook row."""
    action = workbook_case().first
    arguments = {**action.arguments, "deletions": [{"sheet": "Orders", "start": row, "count": 1}]}
    return _reply(
        "world-v1",
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": f"edit-{row}",
                    "function": {"name": action.name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


def _done() -> JsonHttpResponse:
    """One scripted actor reply that finishes."""
    return _reply("world-v1", {"role": "assistant", "content": "done"})


def _tool_reply(name: str, arguments: JsonObject) -> JsonHttpResponse:
    """One scripted role reply making a single forced tool call."""
    return _reply(
        "judge-v1",
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": name,
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


def _spec(project: ProjectStore, job: ContinuationJob, *, judges: int = 1) -> ActorCriticSpec:
    """Register a three-attempt group, one critic and two continuations per point."""
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    settings = AuthorSettings(
        model_alias="judge",
        model=catalog.snapshot("judge")[0],
        catalog_sha256=job.catalog_sha256,
        output_tokens=1024,
        deadline=job.deadline,
    )
    group = job.model_copy(
        update={
            "maximum_physical_attempts": 12,
            "study": job.study.model_copy(update={"repeats": 3}),
        }
    )
    return ActorCriticSpec(
        initial=group, judges=(settings,) * judges, critics=(settings,), points=2, continuations=2
    )


def _point(step: int) -> JsonObject:
    """One estimation point citing only supplied locators."""
    return {
        "step": step,
        "decision": "Delete an order row",
        "alternatives": ["Inspect the sheet first"],
        "deciding_state": [],
        "rationale": "Attempts diverge after this edit.",
        "evidence_refs": [f"point:{step}", "group"],
    }


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    actor: list[JsonHttpResponse],
    roles: list[JsonHttpResponse],
) -> tuple[ScriptedJsonTransport, ScriptedJsonTransport]:
    """Route actor and role calls to separate scripted transports."""
    actor_transport = ScriptedJsonTransport(tuple(actor))
    role_transport = ScriptedJsonTransport(tuple(roles))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor_transport
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: role_transport
    )
    return actor_transport, role_transport


def test_a_run_judges_estimates_continues_and_rewards_then_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every step runs once over real saved owners, and a second call replays the report."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    roles = [
        _tool_reply(
            "choose_points",
            {
                "points": [_point(0), _point(1)],
                "audit": [
                    {
                        "category": "guess_required",
                        "explanation": "The target row is never shown.",
                        "evidence_refs": ["verifier"],
                    }
                ],
                "limitations": ["Simulated grades only."],
            },
        ),
        _tool_reply(
            "estimate_points",
            {
                "estimates": [
                    {
                        "step": 0,
                        "probability": 0.5,
                        "rationale": "Base rate.",
                        "evidence_refs": ["group"],
                    },
                    {
                        "step": 1,
                        "probability": 0.9,
                        "rationale": "Edit done.",
                        "evidence_refs": ["point:1"],
                    },
                ],
                "limitations": ["Simulated grades only."],
            },
        ),
    ]
    actor_transport, role_transport = _wire(monkeypatch, group + continuations, roles)

    report = run_actor_critic(project, spec, code_revision="test")

    assert len(role_transport.requests) == 2
    assert len(actor_transport.requests) == len(group) + len(continuations)
    assert [row.point for row in report.points] == [0, 1]
    assert all(
        row.failure is None and row.outcome.graded + row.outcome.unknown == 2
        for row in report.points
    )
    assert report.judges[0].selection is not None and report.judges[0].score is not None
    assert report.critics[0].assessment is not None and report.critics[0].score is not None
    assert report.audit[0]["category"] == "guess_required"
    assert sorted(report.ranking) == [0, 1]
    for row in report.points:
        assert len(row.group.advantages) == 2

    replay = run_actor_critic(project, spec, code_revision="test")

    assert replay == report
    assert len(role_transport.requests) == 2


def test_a_judge_citing_an_unsupplied_locator_is_kept_as_a_failed_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invented reference never selects points; with no usable judge the run refuses."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    invented = {**_point(1), "evidence_refs": ["attempt:99"]}
    roles = [_tool_reply("choose_points", {"points": [invented], "limitations": ["none"]})]
    _wire(monkeypatch, group, roles)

    with pytest.raises(ValueError, match="no judge produced a usable selection"):
        run_actor_critic(project, spec, code_revision="test")

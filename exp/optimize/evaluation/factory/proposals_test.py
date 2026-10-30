"""Public-context authors retain multiple candidates without seeing private grader data."""

import json
from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.package import export_executable_harbor
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.proposals import (
    TaskProposalSpec,
    preflight_task_proposals,
    proposal_request,
    propose_tasks,
)
from exp.optimize.evaluation.factory.variants import load_task_variant
from exp.optimize.evaluation.factory.variants_test import variant_fixture
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.continuations import load_continuation_checkpoint


def proposal_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, duplicate: bool = False
) -> tuple[ProjectStore, TaskProposalSpec, ScriptedJsonTransport]:
    """Provide two authored prompts through the real provider parser and bounded transport."""
    project, variant = variant_fixture(tmp_path, monkeypatch)
    base = load_executable_task(project, variant.base)
    spec = TaskProposalSpec(
        base=variant.base,
        brief=variant.brief,
        model_alias=base.job.actor_alias,
        model=base.job.study.actor_model,
        catalog_sha256=base.job.catalog_sha256,
        deadline=base.job.deadline,
    )
    prompts = (
        "Delete the first data row; retain the source.",
        "Remove the first data row without overwriting the input.",
    )
    rows = [
        {
            "prompt": prompts[0 if duplicate else i],
            "rationale": f"Alternative {i}; same useful outcome.",
        }
        for i in range(2)
    ]
    response = JsonHttpResponse(
        status_code=200,
        body={
            "id": "proposals",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "proposal",
                                "type": "function",
                                "function": {
                                    "name": "propose_tasks",
                                    "arguments": json.dumps({"candidates": rows}),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )
    transport = ScriptedJsonTransport((response,))
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    return project, spec, transport


def test_public_authoring_keeps_both_candidates_unqualified_and_same_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the prompt varies; world, verifier and provenance remain independently bound."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    preflight_task_proposals(project, spec)
    assert not transport.requests
    request = proposal_request(project, spec)
    public = json.loads(request.messages[1].content or "")
    assert set(public) == {
        "kind",
        "intent",
        "brief",
        "task",
        "messages",
        "tools",
        "remaining_model_calls",
    }
    assert "verifier" not in public and "world" not in public and "outcomes" not in public
    report = propose_tasks(project, spec, code_revision="test")
    assert len(report.variants) == 2 and len(transport.requests) == 1
    base = load_executable_task(project, spec.base)
    source = load_continuation_checkpoint(project.artifacts, base.job.study.checkpoint)
    for reference in report.variants:
        variant = load_task_variant(project, reference)
        task = load_executable_task(project, variant.task)
        point = load_continuation_checkpoint(project.artifacts, variant.checkpoint)
        assert (
            point.world == source.world
            and point.task.lineage_group_id == source.task.lineage_group_id
        )
        assert variant.spec.proposal == report.response
        assert task.job.study.verifier == base.job.study.verifier
        assert task.job.study.training_specification is None
        with pytest.raises(ValueError, match="unqualified"):
            export_executable_harbor(
                project, variant.task, tmp_path / reference.artifact_id, name="test/candidate"
            )
    base.job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    assert propose_tasks(project, spec, code_revision="test") == report
    assert len(transport.requests) == 1


def test_invalid_saved_candidates_never_buy_another_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate prototype rejection preserves the original response instead of regenerating it."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch, duplicate=True)
    for _ in range(2):
        with pytest.raises(ValueError, match="distinct wording"):
            propose_tasks(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")
    assert not project.artifacts.list_ids(artifact_type="task-variant")


def test_unavailable_response_is_not_redrawn_and_capacity_refuses_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capacity and unknown intents are explicit failures, never a request to truncate or retry."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    too_small = spec.model_copy(update={"maximum_input_bytes": 512})
    with pytest.raises(ValueError, match="byte allowance"):
        propose_tasks(project, too_small, code_revision="test")
    assert not transport.requests
    unavailable = ScriptedJsonTransport(())

    def lose_response() -> JsonHttpResponse:
        """Fail after the admitted transport has claimed its physical attempt."""
        raise RuntimeError("lost provider response")

    monkeypatch.setattr(unavailable, "_answer", lose_response)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: unavailable
    )
    with pytest.raises((RuntimeError, ValueError)):
        propose_tasks(project, spec, code_revision="test")
    count = len(unavailable.requests)
    with pytest.raises(ValueError, match="unknown outcome"):
        propose_tasks(project, spec, code_revision="test")
    assert len(unavailable.requests) == count

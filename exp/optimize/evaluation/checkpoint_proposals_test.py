"""Automatic selectors retain source attempts and hand off only checkpoints to future critics."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation import checkpoint_proposals as owner
from exp.optimize.evaluation.candidate_batch_test import _reply
from exp.optimize.evaluation.checkpoint_proposals import CheckpointProposalSpec, propose_checkpoints
from exp.optimize.evaluation.checkpoint_selection import load_selection_plan
from exp.optimize.evaluation.continuation_job import (
    continuation_result_reference,
    execute_continuation_job,
)
from exp.optimize.evaluation.continuation_job_test import job_fixture
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog


def selector_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, choice: int = 1
) -> tuple[ProjectStore, CheckpointProposalSpec, ScriptedJsonTransport]:
    """Create three real native workbook trajectories and a single scripted selector draw."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    job = job.model_copy(
        update={
            "maximum_physical_attempts": 6,
            "study": job.study.model_copy(update={"repeats": 3}),
        }
    )
    action = workbook_case().first
    replies = []
    for row in (2, 3, 4):
        arguments = {
            **action.arguments,
            "deletions": [{"sheet": "Orders", "start": row, "count": 1}],
        }
        replies.extend(
            (
                _reply(
                    "world-v1",
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"edit-{row}",
                                "function": {
                                    "name": action.name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "tool_calls",
                ),
                _reply("world-v1", {"role": "assistant", "content": "done"}),
            )
        )
    actor_transport = ScriptedJsonTransport(tuple(replies))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuation_job.HttpxJsonTransport", lambda: actor_transport
    )
    study = execute_continuation_job(project, job, code_revision="test")
    assert len(study.branches) == 3 and len(actor_transport.requests) == 6
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    spec = CheckpointProposalSpec(
        base=job,
        studies=(continuation_result_reference(project, job, code_revision="test"),),
        selector_alias="judge",
        selector_model=catalog.snapshot("judge")[0],
        catalog_sha256=job.catalog_sha256,
        points=1,
        output_tokens=1024,
        maximum_input_bytes=512000,
        deadline=job.deadline,
        horizon_tolerance=1,
    )
    selector = ScriptedJsonTransport(
        (
            _reply(
                "judge-v1",
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "choose",
                            "function": {
                                "name": "select_points",
                                "arguments": json.dumps(
                                    {
                                        "choices": [
                                            {
                                                "index": choice,
                                                "rationale": "Compare recovery after edit",
                                            }
                                        ],
                                        "limitations": "Source grades are not independent truth",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "tool_calls",
            ),
        )
    )
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: selector)
    return project, spec, selector


def test_complete_source_inventory_and_saved_plan_replay_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrospective draw chooses an actual point and publishes existing matched controls."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    report = propose_checkpoints(project, spec, code_revision="test")
    assert report is not None and report.selection is not None
    assert len(transport.requests) == 1
    plan = load_selection_plan(project, report.selection)
    assert plan.matches[0].control is not None
    assert plan.spec.proposed[0].checkpoint == report.pool[1]
    request = json.loads(
        project.artifacts.read_bytes(spec.identity("test") + "-intent", "request.json")
    )
    packet = json.loads(request["messages"][1]["content"])
    assert len(packet["source_studies"][0]["branches"]) == 3
    assert len(packet["checkpoints"]) == len(report.pool)
    project.model_catalog_path.unlink()
    spec.base.runtime_file.unlink()
    assert propose_checkpoints(project, spec, code_revision="test") == report
    assert len(transport.requests) == 1


def test_invented_point_is_saved_and_never_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An out-of-pool proposal is not mapped to the nearest valid point or silently retried."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch, choice=999)
    for _ in range(2):
        with pytest.raises(ValueError, match="pool/point allowance"):
            propose_checkpoints(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")
    assert not project.artifacts.exists(spec.identity("test"))


def test_unknown_selector_intent_never_rebuys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted selector cannot silently draw a more convenient proposal."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)

    def interrupted() -> ScriptedJsonTransport:
        """Interrupt after durable intent, before any provider call."""
        raise KeyboardInterrupt

    monkeypatch.setattr(owner, "HttpxJsonTransport", interrupted)
    with pytest.raises(KeyboardInterrupt):
        propose_checkpoints(project, spec, code_revision="test")
    with pytest.raises(ValueError, match="unknown outcome"):
        propose_checkpoints(project, spec, code_revision="test")
    assert not transport.requests


def test_cli_selection_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatic proposals use the paid-command admission path and saved results replay freely."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    path = tmp_path / "select.json"
    path.write_text(
        ContinuationCommand(
            action="propose-checkpoints", checkpoint_proposals=spec
        ).model_dump_json()
    )
    args = ["eval", "case", "--root", str(tmp_path), "--continuations", str(path)]
    dry = CliRunner().invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    run = CliRunner().invoke(app, [*args, "--yes", "--non-interactive"])
    assert run.exit_code == 0, run.output
    assert '"selection"' in run.output
    project.model_catalog_path.unlink()
    spec.base.runtime_file.unlink()
    replay = CliRunner().invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == 1


def test_oversized_evidence_refuses_before_selector_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete source inventory cannot be silently shortened to fit the selector."""
    project, spec, transport = selector_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"maximum_input_bytes": 512})
    with pytest.raises(ValueError, match="complete evidence exceeds"):
        propose_checkpoints(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")


def test_truncated_selector_reply_never_publishes_even_when_json_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider truncation is evidence of an incomplete draw, not a usable point selection."""
    project, spec, _ = selector_fixture(tmp_path, monkeypatch)
    transport = ScriptedJsonTransport(
        (
            _reply(
                "judge-v1",
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "choose",
                            "function": {
                                "name": "select_points",
                                "arguments": json.dumps(
                                    {
                                        "choices": [{"index": 1, "rationale": "Recovery"}],
                                        "limitations": "Fixture",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "length",
            ),
        )
    )
    monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: transport)
    for _ in range(2):
        with pytest.raises(ValueError):
            propose_checkpoints(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert project.artifacts.exists(spec.identity("test") + "-response")
    assert not project.artifacts.exists(spec.identity("test"))

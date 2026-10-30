"""Source extraction keeps all public messages and refuses mismatched observations."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactEnvelope, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.common.traces.ingest.dataset import persist_trace_dataset
from exp.optimize.evaluation.equivalence.prefixes import (
    ActorPrefixSpec,
    admit_actor_prefix,
    persist_actor_prefix,
)


def prefix_fixture(tmp_path: Path, fault: str = "") -> tuple[ProjectStore, ActorPrefixSpec]:
    """Import a source with private future outputs and an independently saved simulator reply."""
    project = ProjectStore(tmp_path, "prefix")
    action = {"name": "read", "arguments": {"key": "one"}}
    messages: list[JsonObject] = [
        {"role": "system", "content": "Preserve all other records."},
        {"role": "developer", "content": "Use the declared tools."},
        {"role": "user", "content": "Read record one."},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"key":"one"}'},
                }
            ],
        },
        {"role": "tool", "name": "read", "tool_call_id": "call", "content": "value=12"},
        {"role": "assistant", "content": "PRIVATE FUTURE OUTPUT"},
    ]
    if fault == "media":
        messages[2]["content"] = [{"type": "text", "text": "Read record one."}]
    if fault == "extension":
        messages[0]["private_extension"] = "must not silently discard"
    if fault == "parallel":
        calls = messages[3]["tool_calls"]
        assert isinstance(calls, list)
        calls.append(
            {
                "id": "sibling",
                "type": "function",
                "function": {"name": "read", "arguments": '{"key":"two"}'},
            }
        )
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "id": "trace",
                "messages": messages,
                "metadata": {"prompt": "reconstructed from template"},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read a record",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )
    )
    normalized = CHAT_JSON_SOURCE.load(path)
    trace = normalized.traces[0]
    saved = persist_trace_dataset(
        normalized, project.artifacts, created_at=datetime.now(UTC), code_revision="test"
    )
    candidate_action = {**action, "arguments": {"key": "two"}} if fault == "action" else action
    reference = project.artifacts.write_json(
        artifact_id="evaluation",
        artifact_type="workflow-evaluation-intent",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={
            "evidence.json": {
                "workflow_id": "run",
                "task": trace.task,
                "initial_state": {},
                "steps": [
                    {"step": 0, "action": candidate_action, "status": "valid", "reply": "value=99"}
                ],
                "audit_metadata": {"private": "DO NOT DISCLOSE"},
            },
            "references.json": {
                "capability": "recorded_reference_only",
                "steps": [
                    {
                        "step": 0,
                        "action": action,
                        "reply_available": True,
                        "reply": "other" if fault == "reply" else "value=12",
                    }
                ],
            },
        },
    )
    return project, ActorPrefixSpec(
        dataset=artifact_input(project.artifacts.read(saved.dataset.dataset_id).manifest),
        trace_id=trace.trace_id,
        model_span_id=trace.spans[-1].span_id,
        observation_call_id="call",
        evaluation=artifact_input(reference),
        step=0,
    )


def test_preserves_entire_public_prefix_and_original_source_limits(tmp_path: Path) -> None:
    """Neither hidden workflow fields nor later recorded decisions enter the new actor request."""
    project, spec = prefix_fixture(tmp_path)
    case = admit_actor_prefix(project.artifacts, spec)
    assert [message.role for message in case.request.messages] == [
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
    ]
    assert case.request.messages[-1].content == "value=12"
    assert case.for_arm("simulated").messages[-1].content == "value=99"
    assert "PRIVATE" not in case.request.model_dump_json()
    assert "DO NOT DISCLOSE" not in case.model_dump_json()
    assert "reconstructed from template" in case.admission
    assert "original-context completeness" in case.admission
    first = persist_actor_prefix(
        project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    assert (
        persist_actor_prefix(
            project.artifacts, spec, created_at=datetime.now(UTC), code_revision="test"
        )
        == first
    )


@pytest.mark.parametrize("fault", ["media", "extension", "parallel", "action", "reply", "digest"])
def test_unsupported_or_mismatched_prefix_never_admits(tmp_path: Path, fault: str) -> None:
    """Incomplete turns and lossy parsing fail before a provider could see a request."""
    project, spec = prefix_fixture(tmp_path, fault)
    if fault == "digest":
        spec = spec.model_copy(
            update={"dataset": spec.dataset.model_copy(update={"sha256": "f" * 64})}
        )
    with pytest.raises(ValueError):
        admit_actor_prefix(project.artifacts, spec)


def test_actual_cli_admits_and_replays_without_a_workbook(tmp_path: Path) -> None:
    """Source extraction is an ordinary maintained CLI action, not a benchmark helper."""
    project, spec = prefix_fixture(tmp_path)
    path = tmp_path / "admit.json"
    path.write_text(ContinuationCommand(action="admit-actor-prefix", prefix=spec).model_dump_json())
    args = ["eval", "prefix", "--root", str(tmp_path), "--continuations", str(path)]
    dry = CliRunner().invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="actor-prefix")
    for _ in range(2):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "saved_prefix" in result.output
    assert len(project.artifacts.list_ids(artifact_type="actor-prefix")) == 1

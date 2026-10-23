"""Memory reaches predictions as typed channels, independent of message framing."""

import json
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import AssistantAction, ToolCall
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.tests.fidelity_lifecycle_test import learn_alpha
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    judged,
    memory_project,
    predicted,
    trajectory,
)
from exp.simulation.engines.text.prompt import (
    MEMORY_CONTEXT_GUIDANCE,
    MemoryContext,
    build_world_model_request,
)

OBSERVATION: JsonObject = {
    "id": "consumed-trajectory-x-step-0",
    "kind": "original-observation",
    "observed_reply": "other-instance-reply",
}
SESSION: JsonObject = {"id": "session-note", "session_id": "now", "investigation": None}
PRIOR: JsonObject = {"id": "prior-note", "session_id": "earlier", "investigation": {}}
OVERVIEW: JsonObject = {"id": "#0-1", "kind": "summary", "text": "compressed"}


def render(protocol: Literal["text", "json-tool"]) -> tuple[str, JsonObject]:
    """Render one fixed context under the requested output framing."""
    task = TaskCase(
        task_id="t",
        origin="authored",
        lineage_group_id="l",
        partition="held_out",
        instruction="Inspect",
        tools=(ToolSchema(name="inspect", description="read", input_schema={"type": "object"}),),
        initial_context={"public_prefix": []},
        workload_weight=1,
    )
    request = build_world_model_request(
        task,
        visible_messages=(),
        candidate_response=AssistantAction(
            tool_calls=(ToolCall(call_id="c", name="inspect", arguments={}),)
        ),
        grounded_examples=(),
        maximum_output_tokens=100,
        state={"current": "instance"},
        memory=MemoryContext(
            observations=(OBSERVATION,),
            session=(SESSION,),
            prior_trajectories=(PRIOR,),
            overview=(OVERVIEW,),
        ),
        output_protocol=protocol,
    )
    system, user = request.messages
    return str(system.content), json.loads(str(user.content))


def test_framing_changes_preserve_channels_and_exclusions() -> None:
    """Text and typed-proposal framings carry the same channels with the same authority."""
    text_system, text_payload = render("text")
    tool_system, tool_payload = render("json-tool")
    assert text_system != tool_system
    assert MEMORY_CONTEXT_GUIDANCE in text_system and MEMORY_CONTEXT_GUIDANCE in tool_system
    assert text_payload == tool_payload
    assert text_payload["memory_observations"] == [OBSERVATION]
    research = text_payload["research_history"]
    assert isinstance(research, dict)
    assert research["session"] == [SESSION]
    assert research["prior_trajectories"] == [PRIOR]
    assert research["memory_overview"] == [OVERVIEW]
    assert "original-observation" not in json.dumps(research)
    assert text_payload["environment_state"] == {"current": "instance"}


def test_recorded_request_hash_is_the_dispatched_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory is rendered once: the saved hash identifies the exact provider request."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    learn_alpha(project, base, monkeypatch)
    b = trajectory(base, "traj-b", "beta", "learn")
    wire = RoutedTransport(
        [predicted("traj-b-0", "beta-reply-0"), predicted("traj-b-1", "beta-reply-1"), judged(b)]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, b, code_revision="fixture")
    )
    inputs = [
        json.loads(project.artifacts.read_bytes(i, "record.json"))["prepared"]
        for i in project.artifacts.list_ids(artifact_type="world-operation")
        if i.endswith("-input")
    ]
    assert inputs
    for prepared in inputs:
        assert prepared["evidence"]["request_sha256"] == sha256_json(prepared["body"])
    dispatched = wire.requests[0].payload["messages"]
    assert isinstance(dispatched, list) and isinstance(dispatched[1], dict)
    assert "memory_observations" in json.loads(str(dispatched[1]["content"]))

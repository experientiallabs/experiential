"""The ``learn-agent-memory`` action end to end through ``exp eval --continuations``.

The learning model is a real catalog alias on the Anthropic Messages wire; only the HTTP
transport is scripted, so admission, spend consent, request budgets, replay and export all run.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.learning import agent_memory
from exp.optimize.evaluation.learning.agent_memory import memory_identity
from exp.optimize.evaluation.learning.agent_memory_contracts import AgentMemory
from exp.optimize.evaluation.learning.agent_memory_test import at, project_root
from exp.runtime.models.providers.transport import JsonHttpResponse, RecordedRequest

PROJECT = "memory"
CONTRACT = "Write the report to /out/report.md as markdown."


class ScriptedWire:
    """Scripted provider wire answering each stage's named structured proposal.

    Speaks the Anthropic Messages or the OpenAI-compatible Chat Completions dialect, whichever
    the request uses. Extraction proposes one shared contract per shown trace, merging puts every
    candidate in one group and screening accepts every lesson. ``refuse`` makes one stage answer
    outside its schema.
    """

    def __init__(self, refuse: str | None = None) -> None:
        """Start with no requests; ``refuse`` names the stage that answers malformed."""
        self.requests: list[RecordedRequest] = []
        self.refuse = refuse

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record the request and answer its stage through the submit_proposal tool."""
        self.requests.append(RecordedRequest(url, headers, payload))
        tools = payload["tools"]
        assert isinstance(tools, list)
        chat = url.endswith("/chat/completions")
        schema = at(tools[0], "function", "parameters") if chat else at(tools[0], "input_schema")
        properties = at(schema, "properties")
        assert isinstance(properties, dict)
        stage, answer = self._answer(properties)
        if stage == self.refuse:
            answer = {"unexpected": True}
        if chat:
            call = {"name": "submit_proposal", "arguments": json.dumps(answer)}
            body: JsonObject = {
                "id": f"chat-{len(self.requests)}",
                "model": payload["model"],
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call", "type": "function", "function": call}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        else:
            body = {
                "id": f"msg-{len(self.requests)}",
                "type": "message",
                "role": "assistant",
                "model": payload["model"],
                "content": [
                    {"type": "tool_use", "id": "call", "name": "submit_proposal", "input": answer}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        return JsonHttpResponse(status_code=200, body=body)

    @staticmethod
    def _answer(properties: JsonObject) -> tuple[str, JsonObject]:
        """The stage a response schema belongs to and a valid answer for it."""
        if "screenings" in properties:
            count = at(properties, "screenings", "maxItems")
            assert isinstance(count, int)
            rows = [
                {"lesson": index, "general": 0.9, "covered": 0.0, "supported": 0.9}
                for index in range(count)
            ]
            return "screen", {"screenings": rows}
        item = at(properties, "lessons", "items", "properties")
        assert isinstance(item, dict)
        if "sources" in item:
            last = at(item, "sources", "items", "maximum")
            assert isinstance(last, int)
            group: JsonObject = {
                "statement": CONTRACT,
                "applies_when": "A task asks for a report.",
                "sources": list(range(last + 1)),
            }
            return "merge", {"lessons": [group]}
        labels = at(item, "traces", "items", "enum")
        assert isinstance(labels, list)
        rows = [
            {
                "kind": "task-contract",
                "statement": CONTRACT,
                "applies_when": "A task asks for a report.",
                "evidence": f"{label} wrote /out/report.md.",
                "traces": [label],
            }
            for label in labels
        ]
        return "extract", {"lessons": rows}


def traces_file(tmp_path: Path) -> Path:
    """Three outcome-labelled chat JSON conversations: two passed and one failed."""
    rows = []
    for index, status in enumerate(("success", "success", "failure")):
        outcome: JsonObject = {"exp.outcome.status": status}
        if status == "failure":
            outcome |= {
                "exp.outcome.failure.code": "validation",
                "exp.outcome.failure.message": "verifier failed",
            }
        rows.append(
            {
                "trace_id": f"t{index}",
                **outcome,
                "messages": [
                    {"role": "user", "content": f"Write report {index}."},
                    {"role": "assistant", "content": "Wrote /out/report.md."},
                ],
            }
        )
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def operation(tmp_path: Path) -> JsonObject:
    """The action over the fixture export, writing ``tmp_path/memory``."""
    return {
        "action": "learn-agent-memory",
        "agent_memory": {
            "traces": str(traces_file(tmp_path)),
            "output": str(tmp_path / "memory"),
            "alias": "learner",
        },
    }


def learn(
    root: Path, tmp_path: Path, wire: ScriptedWire, monkeypatch: pytest.MonkeyPatch, *flags: str
) -> str:
    """Run the action over the fixture export and return its flattened output."""
    monkeypatch.setattr(agent_memory, "HttpxJsonTransport", lambda: wire)
    path = tmp_path / "operation.json"
    path.write_text(json.dumps(operation(tmp_path)))
    result = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--continuations", str(path), *flags],
    )
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_learning_writes_screened_memory_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three stages run once; the rerun replays every saved answer and rewrites nothing."""
    root = project_root(tmp_path, monkeypatch)
    wire = ScriptedWire()
    output = learn(root, tmp_path, wire, monkeypatch, "--yes", "--non-interactive")
    assert "3 traces (2 passed, 1 failed, 0 unknown; 0 shown cut), 1 extraction requests" in output
    assert "Extracted 3 candidate lessons (0 requests unusable)" in output
    assert "1 of 1 merged lessons accepted, 0 unscreened, 0 requests unusable" in output
    assert [request.url for request in wire.requests] == [
        "https://api.experientiallabs.ai/v1/chat/completions"
    ] * 3
    assert wire.requests[0].payload["model"] == "glm-5.3"
    memory = (tmp_path / "memory" / "MEMORY.md").read_text()
    assert "3 earlier attempts at similar tasks (2 passed, 1 failed" in memory
    assert "[Task contracts](task-contracts.md): 1 lessons" in memory
    contracts = (tmp_path / "memory" / "task-contracts.md").read_text()
    assert f"## {CONTRACT}" in contracts and "Learned from 3 earlier attempts." in contracts
    assert sorted(path.name for path in (tmp_path / "memory").iterdir()) == [
        "MEMORY.md",
        "task-contracts.md",
    ]

    replay = ScriptedWire()
    again = learn(root, tmp_path, replay, monkeypatch, "--non-interactive")
    assert replay.requests == []
    assert "1 of 1 merged lessons accepted" in again


def test_an_anthropic_learning_model_names_its_proposal_tool_without_sampling_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Messages route sends the named tool beside adaptive thinking and no sampling field."""
    root = project_root(tmp_path, monkeypatch, model="claude-sonnet-5")
    wire = ScriptedWire()
    learn(root, tmp_path, wire, monkeypatch, "--yes", "--non-interactive")
    payload = wire.requests[0].payload
    assert wire.requests[0].url == "https://api.anthropic.com/v1/messages"
    assert payload["tool_choice"] == {"type": "tool", "name": "submit_proposal"}
    assert payload["thinking"] == {"type": "adaptive"}
    assert not {"temperature", "top_p", "top_k"} & set(payload)


def test_a_refused_answer_is_recorded_never_redrawn_and_leaves_its_lessons_unexported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed screening answer leaves the lesson unscreened, and a rerun asks nothing."""
    root = project_root(tmp_path, monkeypatch)
    wire = ScriptedWire(refuse="screen")
    output = learn(root, tmp_path, wire, monkeypatch, "--yes", "--non-interactive")
    assert "0 of 1 merged lessons accepted, 1 unscreened, 1 requests unusable" in output
    assert "unusable screen" in output
    assert "No lesson passed screening." in (tmp_path / "memory" / "MEMORY.md").read_text()
    store = ProjectStore(root, PROJECT)
    (memory_id,) = store.artifacts.list_ids(artifact_type="agent-memory")
    saved = AgentMemory.model_validate_json(store.artifacts.read_bytes(memory_id, "memory.json"))
    assert saved.memory_id == memory_identity(saved.candidates)
    assert [(row.stage, row.status) for row in saved.failed] == [("screen", "rejected")]
    assert saved.lessons[0].screening is None and not saved.lessons[0].accepted
    calls = len(wire.requests)
    learn(root, tmp_path, wire, monkeypatch, "--non-interactive")
    assert len(wire.requests) == calls


def test_dry_run_prints_the_preflight_and_makes_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan is registered and priced, but no credential is used and nothing is written."""
    root = project_root(tmp_path, monkeypatch)
    wire = ScriptedWire()
    output = learn(root, tmp_path, wire, monkeypatch, "--dry-run")
    assert "1 extraction requests with learner, at most $" in output
    assert "Dry run: no provider calls." in output
    assert wire.requests == []
    assert not (tmp_path / "memory").exists()


def test_a_model_that_refuses_named_tool_choice_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structured proposals name their tool; a model that answers that with a 400 is refused."""
    root = project_root(tmp_path, monkeypatch, model="claude-sonnet-5-5")
    wire = ScriptedWire()
    monkeypatch.setattr(agent_memory, "HttpxJsonTransport", lambda: wire)
    path = tmp_path / "operation.json"
    path.write_text(json.dumps(operation(tmp_path)))
    result = CliRunner().invoke(
        app, ["eval", PROJECT, "--root", str(root), "--continuations", str(path), "--yes"]
    )
    assert result.exit_code != 0
    assert "rejects a named tool choice" in " ".join(result.output.split())
    assert wire.requests == []


def test_an_unpriced_learning_model_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request needs a known reservation, so a missing price stops the preflight."""
    root = project_root(tmp_path, monkeypatch, priced=False)
    wire = ScriptedWire()
    monkeypatch.setattr(agent_memory, "HttpxJsonTransport", lambda: wire)
    path = tmp_path / "operation.json"
    path.write_text(json.dumps(operation(tmp_path)))
    result = CliRunner().invoke(
        app, ["eval", PROJECT, "--root", str(root), "--continuations", str(path), "--yes"]
    )
    assert result.exit_code != 0
    assert "has no input or output price" in " ".join(result.output.split())
    assert wire.requests == []

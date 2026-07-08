"""Tests for harness inference from sparse trace evidence."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, HarnessSource, Observation, Step
from wmh.scenarios.mining.facets_test import FakeProvider
from wmh.scenarios.synthesis.infer_harness import (
    harness_evidence,
    infer_harness,
    observed_tools,
)


def _step(
    tool: str,
    args: dict,
    obs: str = "ok",
    *,
    is_error: bool = False,
    task: str = "make me a website",
) -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name=tool, arguments=args),
        observation=Observation(content=obs, is_error=is_error),
        state_before=EnvState(structured={"cwd": "/workspace", "harness": "pi"}),
        task=task,
    )


_STEPS = [
    _step("bash", {"command": "ls"}, "README.md\n"),
    _step("bash", {"command": "mkdir -p /workspace/app", "timeout": 30}, ""),
    _step(
        "write",
        {},
        'Validation failed for tool "write":\n'
        "  - path: must have required properties path, content\n\n"
        "Received arguments:\n{}",
        is_error=True,
    ),
    _step("write", {"path": "/workspace/app.py", "content": "print(1)"}, "Successfully wrote"),
]


def test_observed_tools_infers_schemas_from_calls_and_validation_errors() -> None:
    tools = {t.name: t for t in observed_tools(_STEPS)}
    assert set(tools) == {"bash", "write"}
    bash_properties = tools["bash"].parameters["properties"]
    assert isinstance(bash_properties, dict)
    assert bash_properties["command"] == {"type": "string"}
    assert bash_properties["timeout"] == {"type": "number"}
    # `command` appears in every non-empty bash call; `timeout` doesn't -> not required.
    assert tools["bash"].parameters["required"] == ["command"]
    # The validation error states write's required properties verbatim; honor them exactly.
    assert tools["write"].parameters["required"] == ["path", "content"]


def test_harness_evidence_digest_is_grounded_and_bounded() -> None:
    digest = harness_evidence(_STEPS)
    assert "bash" in digest and "write" in digest
    assert '"command"' in digest  # example calls survive
    assert "must have required properties" in digest  # validation errors survive
    assert "make me a website" in digest  # task samples survive
    assert '"harness":"pi"' in digest or '"harness": "pi"' in digest  # structured state survives


def test_infer_harness_parses_reply_and_marks_source() -> None:
    reply = (
        '{"system_prompt": "You are a coding agent. Tools: bash, write.", '
        '"tools": ['
        '{"name": "bash", "description": "Run a command", "parameters": '
        '{"type": "object", "properties": {"command": {"type": "string"}}, '
        '"required": ["command"]}}, '
        '{"name": "write", "description": "Write a file", "parameters": '
        '{"type": "object", "properties": {"path": {"type": "string"}, '
        '"content": {"type": "string"}}, "required": ["path", "content"]}}, '
        '{"name": "teleport", "description": "not real", "parameters": {}}]}'
    )
    provider = FakeProvider(reply)
    harness = infer_harness(_STEPS, provider)
    assert harness.source is HarnessSource.INFERRED
    assert harness.system_prompt.startswith("You are a coding agent.")
    # A tool never observed in the corpus is dropped, even if the LLM invents it.
    assert [t.name for t in harness.tools] == ["bash", "write"]
    # The evidence digest is what the LLM saw.
    assert provider.last_user is not None and "must have required properties" in provider.last_user


def test_infer_harness_appends_observed_tools_the_reply_missed() -> None:
    reply = (
        '{"system_prompt": "You are an agent.", '
        '"tools": [{"name": "bash", "description": "", "parameters": {}}]}'
    )
    harness = infer_harness(_STEPS, FakeProvider(reply))
    names = [t.name for t in harness.tools]
    assert "write" in names  # observed but missing from the reply -> deterministic schema added
    write = next(t for t in harness.tools if t.name == "write")
    assert write.parameters["required"] == ["path", "content"]


def test_infer_harness_garbage_reply_falls_back_to_observed_schemas() -> None:
    harness = infer_harness(_STEPS, FakeProvider("no json here"))
    assert harness.source is HarnessSource.INFERRED
    assert harness.system_prompt == ""
    assert {t.name for t in harness.tools} == {"bash", "write"}
    assert harness  # tools alone make it usable downstream


def test_infer_harness_message_only_corpus_yields_empty_tools() -> None:
    steps = [
        Step(
            action=Action(kind=ActionKind.MESSAGE, content="hello"),
            observation=Observation(content="hi"),
        )
    ]
    harness = infer_harness(steps, FakeProvider('{"system_prompt": "You chat.", "tools": []}'))
    assert harness.system_prompt == "You chat."
    assert harness.tools == []

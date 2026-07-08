"""Tests for scenario synthesis from a plain task description."""

from __future__ import annotations

from wmh.core.types import (
    Action,
    ActionKind,
    EnvState,
    HarnessContext,
    Observation,
    Step,
    ToolDefinition,
)
from wmh.scenarios.mining.facets_test import FakeProvider
from wmh.scenarios.synthesis.from_task import scenario_from_task

_HARNESS = HarnessContext(
    system_prompt="You are an expert coding assistant operating inside pi.",
    tools=[
        ToolDefinition(
            name="bash",
            description="Run a shell command",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        )
    ],
)


def _example(structured: dict[str, str]) -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
        observation=Observation(content="README.md"),
        state_before=EnvState(structured=dict(structured)),
        task="make me a website",
    )


_EXAMPLES = [
    _example({"cwd": "/workspace", "harness": "pi"}),
    _example({"cwd": "/workspace", "harness": "pi"}),
    _example({"cwd": "/tmp"}),
]


def test_task_is_verbatim_and_synthesis_is_grounded() -> None:
    reply = (
        '{"initial_state": "An empty /workspace directory with python3 available.", '
        '"checklist": ["A runnable python app exists", " ", "Listings can be browsed"]}'
    )
    provider = FakeProvider(reply)
    scenario = scenario_from_task(
        "build a python airbnb clone", _HARNESS, provider, examples=_EXAMPLES
    )

    # The user message stays exactly what the user typed — that's the token-realism contract.
    assert scenario.task == "build a python airbnb clone"
    assert scenario.harness == _HARNESS
    assert scenario.seed_state.scratchpad == "An empty /workspace directory with python3 available."
    # Structured seed state is the modal state of the corpus examples, not LLM-invented.
    assert scenario.seed_state.structured == {"cwd": "/workspace", "harness": "pi"}
    assert scenario.checklist == ["A runnable python app exists", "Listings can be browsed"]
    assert scenario.provenance == []
    assert scenario.weight == 1.0
    # The synthesis prompt is grounded in the harness and real corpus steps.
    assert provider.last_user is not None
    assert "operating inside pi" in provider.last_user
    assert "build a python airbnb clone" in provider.last_user
    assert "README.md" in provider.last_user


def test_scenario_id_is_deterministic_for_the_same_task() -> None:
    reply = '{"initial_state": "", "checklist": ["done"]}'
    a = scenario_from_task("build x", _HARNESS, FakeProvider(reply), examples=[])
    b = scenario_from_task("build x", _HARNESS, FakeProvider(reply), examples=[])
    c = scenario_from_task("build y", _HARNESS, FakeProvider(reply), examples=[])
    assert a.scenario_id == b.scenario_id
    assert a.scenario_id != c.scenario_id
    assert a.scenario_id.startswith("scenario-")


def test_garbage_reply_still_yields_a_usable_scenario() -> None:
    scenario = scenario_from_task("build x", _HARNESS, FakeProvider("nope"), examples=_EXAMPLES)
    assert scenario.task == "build x"
    assert scenario.checklist == []  # verification will flag it: nothing to grade
    assert scenario.seed_state.scratchpad == ""
    assert scenario.seed_state.structured == {"cwd": "/workspace", "harness": "pi"}


def test_render_messages_are_token_realistic() -> None:
    reply = '{"initial_state": "", "checklist": ["done"]}'
    scenario = scenario_from_task(
        "build a python airbnb clone", _HARNESS, FakeProvider(reply), examples=[]
    )
    system, user = scenario.render_messages()
    assert system.startswith("You are an expert coding assistant operating inside pi.")
    assert '"bash"' in system  # tool schemas ride along
    assert user == "build a python airbnb clone"

"""Induced programs as the simulated owner, deferring to the grounded predictor."""

import json
from collections.abc import Iterator

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ToolCall
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.induction.backend import InducedWorldBackend, InducedWorldFactory
from exp.simulation.induction.contracts import InducedCoverage, SandboxLimits
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.induction.tests.fixtures_test import (
    RECORD_FIT_CALLS,
    RECORD_PROGRAMS,
    RECORD_TOOLS,
    decoded,
    grounded_fallback,
    programs,
    record_episode,
    record_manifest,
    task,
)
from exp.simulation.induction.validation import validate_programs


@pytest.fixture(scope="module")
def sandbox() -> Iterator[ProgramSandbox]:
    """Share one worker across this module's worlds."""
    with ProgramSandbox() as shared:
        yield shared


@pytest.fixture(scope="module")
def coverage(sandbox: ProgramSandbox) -> InducedCoverage:
    """Accept the ticket programs on the native fit episode."""
    report = validate_programs(
        programs(RECORD_PROGRAMS),
        (record_episode("fit-1", RECORD_FIT_CALLS),),
        record_manifest(),
        sandbox,
    )
    assert report.coverage.induced_tools
    return report.coverage


def _world(
    coverage: InducedCoverage,
    sandbox: ProgramSandbox,
    proposals: list[tuple[str, JsonValue, list[JsonObject]]],
) -> tuple[ExecutableWorldRuntime, list[JsonObject]]:
    """Compose induced programs over the real grounded predictor with scripted proposals."""
    initial = record_episode("start", (("count_open", {}),)).initial_state
    fallback, _, requests = grounded_fallback(
        record_manifest(), task(RECORD_TOOLS), initial, proposals
    )
    factory = InducedWorldFactory(coverage, fallback, sandbox)
    return ExecutableWorldRuntime(record_manifest(), factory, owner="simulated"), requests


def _call(name: str, call_id: str, **arguments: JsonValue) -> ToolCall:
    """Build one tool call."""
    return ToolCall(name=name, call_id=call_id, arguments=arguments)


HELD_OUT: tuple[tuple[str, JsonObject], ...] = (
    ("open_ticket", {"title": "monitor"}),
    ("get_ticket", {"id": 2}),
    ("close_ticket", {"id": 2}),
    ("count_open", {}),
    ("get_ticket", {"id": 5}),
)


def test_induced_tools_reproduce_native_replies_without_any_prediction(
    coverage: InducedCoverage, sandbox: ProgramSandbox
) -> None:
    """Held-out calls inside the validated domain match the native world exactly."""
    native = record_episode("held-out", HELD_OUT)
    runtime, requests = _world(coverage, sandbox, [])
    with runtime.open(task(RECORD_TOOLS)) as world:
        for index, (step, (name, arguments)) in enumerate(zip(native.steps, HELD_OUT, strict=True)):
            observation = world.execute(_call(name, f"h{index}", **arguments))
            assert (decoded(observation.content), observation.is_error) == (
                step.reply,
                step.is_error,
            )
            assert observation.metadata["transition"] == "induced"
        assert world.last_execution is not None and world.last_execution.owner == "simulated"
    assert requests == []


def test_predicted_tools_and_out_of_domain_reads_go_to_the_predictor(
    coverage: InducedCoverage, sandbox: ProgramSandbox
) -> None:
    """A tool without a program and a call with unvalidated arguments are predicted."""
    note: list[JsonObject] = [
        {"op": "append", "path": ["domains", "notes", "rows"], "value": ["hello"]}
    ]
    runtime, requests = _world(
        coverage, sandbox, [("n", "noted", note), ("v", {"id": 1, "verbose": True}, [])]
    )
    with runtime.open(task(RECORD_TOOLS)) as world:
        noted = world.execute(_call("add_note", "n", text="hello"))
        assert (noted.content, noted.metadata["transition"]) == ("noted", "predicted")
        verbose = world.execute(_call("get_ticket", "v", id=1, verbose=True))
        assert verbose.metadata["transition"] == "predicted"
        induced = world.execute(_call("count_open", "c"))
        assert induced.metadata["transition"] == "induced"
        backend = world._backend
        assert isinstance(backend, InducedWorldBackend)
        assert "count_open" in backend.induced_tools
    assert len(requests) == 2
    history = json.dumps(requests[1])
    assert "hello" in history


def test_an_unvalidated_write_path_hands_its_domains_to_the_predictor(
    coverage: InducedCoverage, sandbox: ProgramSandbox
) -> None:
    """Closing a closed ticket was never validated, so tickets move to the predictor for good."""
    runtime, requests = _world(
        coverage,
        sandbox,
        [("again", "ticket 1 is already closed", []), ("count", {"open": 0}, [])],
    )
    with runtime.open(task(RECORD_TOOLS)) as world:
        assert (
            world.execute(_call("close_ticket", "first", id=1)).metadata["transition"] == "induced"
        )
        again = world.execute(_call("close_ticket", "again", id=1))
        assert again.metadata["transition"] == "predicted"
        saved = world.checkpoint()
        assert world.execute(_call("count_open", "count")).metadata["transition"] == "predicted"
    assert len(requests) == 2
    with runtime.restore(task(RECORD_TOOLS), saved) as resumed:
        assert resumed.checkpoint() == saved
        backend = resumed._backend
        assert isinstance(backend, InducedWorldBackend) and backend.induced_tools == frozenset()


def test_checkpoints_restore_the_exact_induced_state_and_partition(
    coverage: InducedCoverage, sandbox: ProgramSandbox
) -> None:
    """A restored fork continues identically to the original world."""
    runtime, requests = _world(coverage, sandbox, [])
    with runtime.open(task(RECORD_TOOLS)) as world:
        world.execute(_call("open_ticket", "o", title="desk"))
        saved = world.checkpoint()
        original = world.execute(_call("get_ticket", "g", id=2))
    with runtime.restore(task(RECORD_TOOLS), saved) as resumed:
        assert resumed.checkpoint() == saved
        assert resumed.execute(_call("get_ticket", "g", id=2)) == original
    assert requests == []


def test_the_sandbox_must_use_the_limits_the_programs_were_validated_under(
    coverage: InducedCoverage,
) -> None:
    """A different step budget changes which inputs are in domain, so it is refused."""
    fallback, _, _ = grounded_fallback(
        record_manifest(),
        task(RECORD_TOOLS),
        record_episode("s", (("count_open", {}),)).initial_state,
        [],
    )
    with ProgramSandbox(SandboxLimits(maximum_steps=1000)) as other:
        with pytest.raises(ValueError, match="other sandbox limits"):
            InducedWorldFactory(coverage, fallback, other)

"""The bounded worker process: limits, deadlines, restarts and protocol checks."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.simulation.induction.contracts import SandboxLimits, TransitionProgram
from exp.simulation.induction.sandbox import ProgramSandbox

APPEND = TransitionProgram(
    tool="append",
    source=(
        "def transition(state, args, facts):\n"
        "    state['items'].append(args['value'])\n"
        "    return {'size': len(state['items']), 'seen': facts.get('seen')}\n"
    ),
)


def _program(body: str) -> TransitionProgram:
    """Wrap one return expression or statement block as a program."""
    return TransitionProgram(tool="fixture", source=f"def transition(state, args, facts):\n{body}")


def test_a_program_runs_out_of_process_and_returns_reply_state_and_lines() -> None:
    """The worker executes the call on detached state and the caller's values are untouched."""
    state: JsonObject = {"items": [1]}
    with ProgramSandbox() as sandbox:
        outcome = sandbox.run(APPEND, state, {"value": 2}, {"seen": True})
    assert outcome.status == "reply"
    assert outcome.reply == {"size": 2, "seen": True}
    assert outcome.state == {"items": [1, 2]}
    assert state == {"items": [1]}
    assert outcome.lines == (2, 3)


def test_a_wall_clock_overrun_kills_the_worker_and_the_next_call_gets_a_fresh_one() -> None:
    """A C-level loop that never yields a line event is stopped by the request deadline."""
    with ProgramSandbox(SandboxLimits(request_timeout_seconds=0.5)) as sandbox:
        slow = sandbox.run(_program("    return sum(range(10 ** 12))\n"), {}, {}, {})
        assert slow.status == "failed" and "wall-clock" in str(slow.detail)
        assert sandbox.run(APPEND, {"items": []}, {"value": 1}, {}).status == "reply"


def test_memory_and_file_access_are_contained_by_the_worker_limits() -> None:
    """Huge allocations fail inside the address-space limit and no file can be opened."""
    with ProgramSandbox() as sandbox:
        memory = sandbox.run(_program("    return [0] * (10 ** 10)\n"), {}, {}, {})
        assert memory.status == "failed" and "MemoryError" in str(memory.detail)
        reading = sandbox.run(_program("    return open('/etc/hostname').read()\n"), {}, {}, {})
        assert reading.status == "failed"


def test_refused_programs_never_reach_the_worker() -> None:
    """The static check runs in the parent first."""
    with ProgramSandbox() as sandbox:
        outcome = sandbox.run(_program("    return state.__dict__\n"), {}, {}, {})
        assert outcome.status == "refused"
        assert sandbox._process is None


def test_set_iteration_order_is_reproducible_across_workers() -> None:
    """Hashing is not randomized, so two workers order the same set identically."""
    program = _program("    return list({'alpha', 'beta', 'gamma', 'delta'})\n")
    replies = []
    for _ in range(2):
        with ProgramSandbox() as sandbox:
            replies.append(sandbox.run(program, {}, {}, {}).reply)
    assert replies[0] == replies[1]


def test_a_closed_sandbox_refuses_requests() -> None:
    """Use after close is an explicit error, not a silent restart."""
    sandbox = ProgramSandbox()
    sandbox.close()
    with pytest.raises(RuntimeError, match="closed"):
        sandbox.run(APPEND, {"items": []}, {"value": 1}, {})


def test_non_json_replies_are_failed_outcomes() -> None:
    """A reply that JSON cannot carry is a failure, never a coerced value."""
    with ProgramSandbox() as sandbox:
        outcome = sandbox.run(_program("    return float('nan')\n"), {}, {}, {})
    assert outcome.status == "failed" and "unencodable" in str(outcome.detail)

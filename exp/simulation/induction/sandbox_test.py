"""The bounded worker process: limits, deadlines, restarts and protocol checks."""

import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.simulation.induction import runner
from exp.simulation.induction import sandbox as sandbox_module
from exp.simulation.induction.contracts import SandboxLimits, TransitionProgram
from exp.simulation.induction.runner_test import COUNTER, SECRET, formatter_escape, secret_file
from exp.simulation.induction.sandbox import ProgramSandbox, SandboxUnavailableError

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


def test_memory_and_file_access_are_contained_by_the_worker_limits(tmp_path: Path) -> None:
    """Huge allocations fail inside the address-space limit and no file can be opened."""
    with ProgramSandbox() as sandbox:
        memory = sandbox.run(_program("    return [0] * (10 ** 10)\n"), {}, {}, {})
        assert memory.status == "failed" and "MemoryError" in str(memory.detail)
        path = secret_file(tmp_path)
        reading = sandbox.run(_program(f"    return open({str(path)!r}).read()\n"), {}, {}, {})
        assert reading.status == "failed" and SECRET not in reading.model_dump_json()


def test_the_formatter_escape_reaches_no_host_file(tmp_path: Path) -> None:
    """The reproduced string-driven escape fails in the worker instead of reading a host file."""
    escape = TransitionProgram(tool="fixture", source=formatter_escape(secret_file(tmp_path)))
    with ProgramSandbox() as sandbox:
        outcome = sandbox.run(escape, {}, {}, {})
    assert outcome.status == "failed" and SECRET not in outcome.model_dump_json()


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


def test_one_worker_gives_repeated_and_interleaved_requests_identical_results() -> None:
    """A long-lived worker keeps no program state between requests or across programs."""
    counter = TransitionProgram(tool="counter", source=COUNTER)
    dumps = TransitionProgram(
        tool="dumps",
        source=(
            "import json\ndef transition(state, args, facts):\n    return json.dumps([1, 2, 3])\n"
        ),
    )
    with ProgramSandbox() as sandbox:
        replies = []
        for _ in range(3):
            replies.append(sandbox.run(counter, {}, {}, {}).reply)
            replies.append(sandbox.run(dumps, {}, {}, {}).reply)
    assert replies == [1, "[1, 2, 3]"] * 3


def _busy_size(cpu_seconds: float) -> int:
    """Return a ``sum(range(n))`` size that spends about ``cpu_seconds`` of CPU on this host."""
    size = 10**7
    start = time.process_time()
    sum(range(size))
    return int(size * cpu_seconds / max(time.process_time() - start, 1e-3))


def _busy(cpu_seconds: float) -> TransitionProgram:
    """Return a program whose every request spends about ``cpu_seconds`` of CPU."""
    return _program(f"    return sum(range({_busy_size(cpu_seconds)})) > 0\n")


def _contained_run(script: str, *arguments: str) -> str:
    """Run one sandbox script in a separate interpreter, so its process limits stay there."""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        (sys.executable, "-c", script, *arguments),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        cwd=Path(__file__).parents[3],
    )
    return result.stdout.strip()


def test_the_cpu_limit_applies_to_each_request_not_the_worker_lifetime() -> None:
    """Requests that each fit the limit all reply, though together they use several times it."""
    limits = SandboxLimits(cpu_seconds=1, request_timeout_seconds=30.0)
    busy = _busy(0.4)
    with ProgramSandbox(limits) as sandbox:
        outcomes = [sandbox.run(busy, {}, {}, {}) for _ in range(8)]
    assert [outcome.status for outcome in outcomes] == ["reply"] * 8


_SANDBOX_SCRIPT = """
import math, resource, signal, sys
from exp.simulation.induction.contracts import SandboxLimits, TransitionProgram
from exp.simulation.induction import runner
from exp.simulation.induction import sandbox as sandbox_module
from exp.simulation.induction.sandbox import ProgramSandbox, SandboxUnavailableError

mode, size, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if mode == "hard":
    usage = resource.getrusage(resource.RUSAGE_SELF)
    hard = math.ceil(usage.ru_utime + usage.ru_stime) + 4
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
else:
    signal.signal(signal.SIGXCPU, signal.SIG_IGN)
source = f"def transition(state, args, facts):\\n    return sum(range({size})) > 0\\n"
busy = TransitionProgram(tool="busy", source=source)
workers = set()
with ProgramSandbox(SandboxLimits(cpu_seconds=1, request_timeout_seconds=60.0)) as sandbox:
    for _ in range(count):
        status = sandbox.run(busy, {}, {}, {}).status
        if sandbox._process is not None:
            workers.add(sandbox._process.pid)
        print(status)
print(len(workers) > 1)
"""


def test_a_finite_inherited_hard_limit_rotates_workers_instead_of_failing_requests() -> None:
    """Under a four-second hard CPU limit, twelve half-second requests all reply.

    Each worker can serve only a few of them, so the sandbox replaces a worker before its
    remaining allowance falls below one full request instead of letting the kernel kill it.
    """
    lines = _contained_run(_SANDBOX_SCRIPT, "hard", str(_busy_size(0.5)), "12").splitlines()
    assert lines == ["reply"] * 12 + ["True"]


def test_an_inherited_ignored_sigxcpu_cannot_defeat_the_cpu_limit() -> None:
    """The worker restores the default SIGXCPU action, so a request over its budget stops."""
    lines = _contained_run(_SANDBOX_SCRIPT, "ignored", str(_busy_size(3.0)), "1").splitlines()
    assert lines[0] == "failed"


_ADMISSION_SCRIPT = """
import functools, resource, subprocess, sys
from exp.simulation.induction import sandbox
from exp.simulation.induction.contracts import SandboxLimits, TransitionProgram

hard = int(sys.argv[1])
limit = functools.partial(resource.setrlimit, resource.RLIMIT_CPU, (hard, hard))
sandbox.subprocess.Popen = functools.partial(subprocess.Popen, preexec_fn=limit)
source = "def transition(state, args, facts):\\n    return 1\\n"
program = TransitionProgram(tool="one", source=source)
try:
    with sandbox.ProgramSandbox(SandboxLimits(cpu_seconds=1)) as contained:
        print(contained.run(program, {}, {}, {}).status)
except sandbox.SandboxUnavailableError as error:
    print("unavailable:", error)
"""


def test_a_fresh_worker_is_admitted_when_its_hard_limit_fits_exactly_one_request() -> None:
    """A two-second hard limit fits one one-second budget at startup, so the worker runs it.

    A one-second limit never runs the program. The worker usually refuses it by name, but its
    startup imports count against the limit too, and a slow host can spend the whole second
    there, so the kernel stops the worker first and the sandbox reports the generic exit.
    """
    assert _contained_run(_ADMISSION_SCRIPT, "2") == "reply"
    refused = _contained_run(_ADMISSION_SCRIPT, "1")
    assert refused.startswith("unavailable:")
    assert "hard CPU limit" in refused or "a CPU or memory limit may have been hit" in refused


_FAKE_WORKER = """
import json, pathlib, sys
here = pathlib.Path(__file__).parent
starts = here / "starts"
starts.write_text(str(int(starts.read_text() or 0) + 1) if starts.exists() else "1")
replies = json.loads((here / "replies.json").read_text())
print(json.dumps({"ready": True, "protocol": PROTOCOL}), flush=True)
for line in sys.stdin:
    reply = replies.pop(0) if replies else None
    (here / "replies.json").write_text(json.dumps(replies))
    if reply is None:
        sys.exit(0)
    print(reply, flush=True)
"""


def _fake_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: Sequence[str | None]
) -> ProgramSandbox:
    """Return a sandbox whose workers answer requests with ``replies`` in order.

    A ``None`` reply makes the worker exit without answering, as a lost worker does.
    """
    worker = tmp_path / "worker.py"
    worker.write_text(_FAKE_WORKER.replace("PROTOCOL", repr(runner.RUNNER_PROTOCOL)))
    (tmp_path / "replies.json").write_text(json.dumps(list(replies)))
    monkeypatch.setattr(sandbox_module, "_RUNNER", worker)
    return ProgramSandbox()


def _starts(tmp_path: Path) -> int:
    """Return how many fake workers have started."""
    return int((tmp_path / "starts").read_text())


_REPLY = json.dumps(
    {"outcome": {"status": "reply", "reply": 1, "state": {}, "lines": [2]}, "rotate": False}
)


def test_an_unstarted_request_runs_once_on_a_fresh_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that read but never ran the request is replaced and the request sent once more."""
    with _fake_sandbox(tmp_path, monkeypatch, ['{"unstarted": true}', _REPLY]) as sandbox:
        outcome = sandbox.run(APPEND, {"items": []}, {"value": 1}, {})
    assert outcome.status == "reply" and _starts(tmp_path) == 2


def test_a_second_unstarted_answer_makes_the_sandbox_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The move to a fresh worker happens once; a fresh worker that cannot start is an error."""
    replies = ['{"unstarted": true}', '{"unstarted": true}']
    with _fake_sandbox(tmp_path, monkeypatch, replies) as sandbox:
        with pytest.raises(SandboxUnavailableError, match="did not start"):
            sandbox.run(APPEND, {"items": []}, {"value": 1}, {})
    assert _starts(tmp_path) == 2


def test_a_deeply_nested_ready_line_makes_the_sandbox_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ready line nested past the decoder's recursion limit is a worker that did not start."""
    worker = tmp_path / "worker.py"
    worker.write_text('print("[" * 100_000 + "]" * 100_000, flush=True)\n')
    monkeypatch.setattr(sandbox_module, "_RUNNER", worker)
    with ProgramSandbox() as sandbox:
        with pytest.raises(SandboxUnavailableError, match="did not start"):
            sandbox.run(APPEND, {"items": []}, {"value": 1}, {})


def test_a_lost_response_is_failed_and_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that exits mid-request fails that request; only the next request restarts it."""
    with _fake_sandbox(tmp_path, monkeypatch, [None, _REPLY]) as sandbox:
        lost = sandbox.run(APPEND, {"items": []}, {"value": 1}, {})
        assert lost.status == "failed" and _starts(tmp_path) == 1
        assert sandbox.run(APPEND, {"items": []}, {"value": 1}, {}).status == "reply"
    assert _starts(tmp_path) == 2


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        '{"unstarted": 1}',
        '{"unstarted": true, "extra": 1}',
        json.dumps({"outcome": {"status": "reply", "reply": 1, "lines": []}, "rotate": 1}),
        json.dumps({"outcome": {"status": "reply", "reply": 1, "lines": []}, "rotate": "yes"}),
        json.dumps({"outcome": {"status": "nonsense"}, "rotate": False}),
        json.dumps({"status": "reply", "reply": 1, "lines": []}),
        "[" * 100_000 + "]" * 100_000,
    ],
    ids=lambda reply: reply if len(reply) < 80 else "deeply nested",
)
def test_a_protocol_violation_fails_the_request_and_stops_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    """Anything but the two exact envelopes, with a Boolean ``rotate``, discards the worker."""
    with _fake_sandbox(tmp_path, monkeypatch, [reply]) as sandbox:
        outcome = sandbox.run(APPEND, {"items": []}, {"value": 1}, {})
        assert outcome.status == "failed" and "malformed" in str(outcome.detail)
        assert sandbox._process is None
    assert _starts(tmp_path) == 1

"""Owned subprocess output drains before pipe close; startup failures release the same owner."""

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from exp.runtime.gateway import latency_measure as measure


def test_shutdown_waits_for_sole_output_reader_before_closing_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader paused at teardown cannot race a second communicate reader or a closed file."""
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import signal,sys,time; "
            "signal.signal(signal.SIGINT,lambda *_:(print('final',flush=True),sys.exit(0))); "
            "print('ready',flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "ready"
    entered, release = threading.Event(), threading.Event()
    original = measure._pump_process_output

    def paused(process: subprocess.Popen[str], lines: list[str]) -> None:
        entered.set()
        assert release.wait(timeout=5), "test did not release its reader"
        original(process, lines)

    def no_communicate(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stdout already has one reader; communicate must not read or close it")

    monkeypatch.setattr(measure, "_pump_process_output", paused)
    monkeypatch.setattr(child, "communicate", no_communicate)
    owner = measure.GatewayProcess(child)
    errors: list[Exception] = []

    def close() -> None:
        try:
            measure.stop_gateway_process(owner)
        except Exception as error:  # noqa: BLE001 - deliver thread errors to the test assertion.
            errors.append(error)

    closer = threading.Thread(target=close)
    try:
        assert entered.wait(timeout=5)
        closer.start()
        assert child.wait(timeout=5) == 0
        assert closer.is_alive(), "shutdown must join the paused reader"
        assert not child.stdout.closed, "the pipe must remain open until its reader exits"
    finally:
        release.set()
        closer.join(timeout=5)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    assert not closer.is_alive() and not owner.output_thread.is_alive()
    assert not errors
    assert owner.lines == ["final\n"]
    assert child.stdout.closed
    owner.close()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_startup_failure_stops_and_drains_owned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Readiness failure preserves its exception and releases the child and output pipe."""
    original = subprocess.Popen
    children: list[subprocess.Popen[str]] = []

    def launch(*_args: object, **_kwargs: object) -> subprocess.Popen[str]:
        child = original(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('starting',flush=True); time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        children.append(child)
        return child

    def not_ready(*_args: object, **_kwargs: object) -> None:
        raise failure("injected readiness failure")

    monkeypatch.setattr(measure.subprocess, "Popen", launch)
    monkeypatch.setattr(measure, "wait_for_http_ok", not_ready)
    with pytest.raises(failure, match="injected readiness failure"):
        measure.start_gateway_process(root=tmp_path, port=1, credential="fixture")
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdout is not None and children[0].stdout.closed


def test_output_reader_failure_is_reported_by_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner surfaces reader errors after cleanup instead of hiding thread exceptions."""
    child = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE, text=True)

    def broken_reader(_process: subprocess.Popen[str], _lines: list[str]) -> None:
        raise ValueError("injected reader failure")

    monkeypatch.setattr(measure, "_pump_process_output", broken_reader)
    owner = measure.GatewayProcess(child)
    owner.output_thread.join(timeout=5)
    with pytest.raises(RuntimeError, match="output reader failed") as failure:
        owner.close()
    assert isinstance(failure.value.__cause__, ValueError)
    assert child.poll() is not None
    assert child.stdout is not None and child.stdout.closed
    assert not owner.output_thread.is_alive()


def test_reader_start_failure_rolls_back_child_and_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread that never starts still leaves no child or captured stdout behind."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def refuse_start(_thread: threading.Thread) -> None:
        raise RuntimeError("injected output thread startup failure")

    monkeypatch.setattr(threading.Thread, "start", refuse_start)
    with pytest.raises(RuntimeError, match="injected output thread startup failure"):
        measure.GatewayProcess(child)
    assert child.poll() is not None
    assert child.stdout is not None and child.stdout.closed

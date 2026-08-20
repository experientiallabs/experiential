"""Tests for the bounded local JSONL executable-environment runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments import local as local_module
from exp.runtime.environments.local import (
    LocalProcessCleanupError,
    LocalProcessCrashError,
    LocalProcessEnvironmentRuntime,
    LocalProcessLimits,
    LocalProcessProtocolError,
    _ProcessIdentity,
    _ProcessRecord,
    _signal_exact_identity,
)

_DARWIN_ONLY = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="process-containment integration requires Darwin kqueue fork notifications",
)


@_DARWIN_ONLY
def test_local_process_environment_executes_in_an_ephemeral_workspace(tmp_path: Path) -> None:
    """Each episode gets a private workspace that disappears after simulator-owned cleanup."""
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        observation = session.execute(ToolCall(call_id="call-1", name="workspace"))
        workspace = Path(str(observation.metadata["workspace"]))
        assert workspace.is_dir()
        assert (workspace / "tool-output.txt").read_text(encoding="utf-8") == "written"

    assert not workspace.exists()
    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_DARWIN_ONLY
def test_local_process_environment_times_out_and_cleans_up(tmp_path: Path) -> None:
    """A blocked tool response reaches the configured bound instead of hanging a rollout."""
    runtime = _runtime(
        tmp_path,
        limits=LocalProcessLimits(
            request_timeout_seconds=0.5,
            session_timeout_seconds=1.0,
            cleanup_timeout_seconds=1.0,
        ),
    )

    with runtime.open(_task()) as session:
        with pytest.raises(TimeoutError, match="response exceeded"):
            session.execute(ToolCall(call_id="call-1", name="sleep"))

    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_DARWIN_ONLY
def test_local_process_environment_records_a_child_crash_without_stderr_leakage(
    tmp_path: Path,
) -> None:
    """A child exit becomes a typed environment failure and still removes its workspace."""
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        with pytest.raises(LocalProcessCrashError, match="return code 7"):
            session.execute(ToolCall(call_id="call-1", name="crash"))

    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_DARWIN_ONLY
def test_local_process_environment_rejects_an_oversized_or_unframed_response(
    tmp_path: Path,
) -> None:
    """A malicious output line cannot consume unbounded memory or masquerade as an observation."""
    runtime = _runtime(
        tmp_path,
        limits=LocalProcessLimits(
            request_timeout_seconds=1.0,
            session_timeout_seconds=2.0,
            cleanup_timeout_seconds=1.0,
            maximum_output_bytes=32,
        ),
    )

    with runtime.open(_task()) as session:
        with pytest.raises(LocalProcessProtocolError, match="maximum_output_bytes"):
            session.execute(ToolCall(call_id="call-1", name="oversized"))

    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_DARWIN_ONLY
def test_local_process_environment_rejects_malformed_json_and_cleans_up(tmp_path: Path) -> None:
    """Malformed child output remains protocol failure evidence and leaves no workspace."""
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        with pytest.raises(LocalProcessProtocolError, match="not valid JSON"):
            session.execute(ToolCall(call_id="call-1", name="malformed"))

    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_DARWIN_ONLY
def test_local_process_child_receives_only_the_documented_environment_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic child dump proves parent credentials cannot cross the process boundary."""
    monkeypatch.setenv("EXP_PARENT_SECRET", "must-not-cross")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-either")
    runtime = _runtime(tmp_path)

    with runtime.open(_task()) as session:
        observation = session.execute(ToolCall(call_id="call-1", name="environment"))

    keys = observation.metadata["keys"]
    assert isinstance(keys, list)
    assert set(keys).difference({"__CF_USER_TEXT_ENCODING"}) == {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "TZ",
        "EXP_SANDBOX_WORKSPACE",
    }
    assert "EXP_PARENT_SECRET" not in keys
    assert "OPENAI_API_KEY" not in keys


@_DARWIN_ONLY
def test_local_process_cleanup_kills_orphaned_grandchild_after_leader_exit(
    tmp_path: Path,
) -> None:
    """A fork is identity-cleaned where observed but still fails containment proof closed."""
    runtime = _runtime(tmp_path)

    with pytest.raises(LocalProcessCleanupError, match=r"fork observed from root pid=\d+"):
        with runtime.open(_task()) as session:
            observation = session.execute(ToolCall(call_id="call-1", name="orphan-grandchild"))
            raw_grandchild_pid = observation.metadata["grandchild_pid"]
            assert isinstance(raw_grandchild_pid, int)
            grandchild_pid = raw_grandchild_pid
            time.sleep(0.05)

    if _pid_is_running(grandchild_pid):
        os.kill(grandchild_pid, 9)
    deadline = time.monotonic() + 2.0
    while _pid_is_running(grandchild_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _pid_is_running(grandchild_pid)
    assert list(tmp_path.glob("exp-sandbox-*")) == []


@_DARWIN_ONLY
def test_cleanup_kills_setsid_descendant_without_touching_unrelated_process(
    tmp_path: Path,
) -> None:
    """Immediate setsid escape fails closed without pipe hang or unrelated process signaling."""
    cleanup_timeout = 0.4
    runtime = _runtime(
        tmp_path,
        limits=LocalProcessLimits(
            request_timeout_seconds=1.0,
            session_timeout_seconds=2.0,
            cleanup_timeout_seconds=cleanup_timeout,
        ),
    )
    unrelated = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    grandchild_pid: int | None = None
    try:
        started = time.monotonic()
        with pytest.raises(LocalProcessCleanupError) as raised:
            with runtime.open(_task()) as session:
                observation = session.execute(
                    ToolCall(call_id="call-detached", name="detached-grandchild")
                )
                raw_pid = observation.metadata["grandchild_pid"]
                assert isinstance(raw_pid, int)
                grandchild_pid = raw_pid
        assert time.monotonic() - started <= cleanup_timeout + 0.25
        assert "fork observed from root pid=" in str(raised.value)
        assert unrelated.poll() is None
        assert not any(
            thread.name.startswith(("exp-local-descendants-", "exp-local-environment-"))
            for thread in threading.enumerate()
        )
    finally:
        if grandchild_pid is not None and _pid_is_running(grandchild_pid):
            os.kill(grandchild_pid, 9)
        unrelated.terminate()
        unrelated.wait(timeout=2.0)


def test_reused_pid_identity_is_never_signaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale PID or former group member cannot authorize a signal after PID reuse."""
    stale = _ProcessIdentity(pid=12345, started=(100, 1))
    replacement = _ProcessRecord(
        identity=_ProcessIdentity(pid=12345, started=(200, 2)),
        parent_pid=1,
        zombie=False,
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(local_module, "_process_snapshot", lambda: {12345: replacement})
    monkeypatch.setattr(local_module.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    _signal_exact_identity(stale, 9)

    assert signals == []


def test_local_process_rejects_unsafe_environment_keys() -> None:
    """Configuration cannot reintroduce credentials on any platform."""
    with pytest.raises(ValueError, match="outside the safe allowlist"):
        LocalProcessEnvironmentRuntime(
            (sys.executable, "fixture.py"),
            environment={"API_KEY": "secret"},
        )


@_DARWIN_ONLY
def test_local_process_rejects_symlinked_workspace_parent(tmp_path: Path) -> None:
    """Darwin execution cannot redirect its ephemeral workspace through a symlink."""
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    runtime = LocalProcessEnvironmentRuntime(
        (sys.executable, "fixture.py"),
        workspace_parent=linked_parent,
    )
    with pytest.raises(LocalProcessProtocolError, match="real directory"):
        with runtime.open(_task()):
            pass


def test_local_process_limits_reject_invalid_values_before_launch() -> None:
    """The local resource contract fails before it can start a user executable."""
    with pytest.raises(ValueError, match="cannot exceed"):
        LocalProcessLimits(request_timeout_seconds=2.0, session_timeout_seconds=1.0)
    with pytest.raises(ValueError, match="positive integer"):
        LocalProcessLimits(maximum_output_bytes=0)


def test_unsupported_platform_fails_before_customer_or_worker_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public context preflight allocates no gate, process, pipe, or worker resources."""
    marker = tmp_path / "customer-executed"
    runtime = LocalProcessEnvironmentRuntime(
        (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
        workspace_parent=tmp_path,
    )
    monkeypatch.setattr(local_module.sys, "platform", "unsupported-test")
    before_fds = _open_file_descriptors()
    before_threads = _local_worker_threads()
    started = time.monotonic()
    context = runtime.open(_task())

    with pytest.raises(LocalProcessCleanupError, match="requires Darwin"):
        context.__enter__()

    assert time.monotonic() - started < 0.25
    assert not marker.exists()
    session = context.__dict__["_session"]
    assert session.__dict__["_process"] is None
    assert _open_file_descriptors() == before_fds
    assert _local_worker_threads() == before_threads


def test_darwin_without_kqueue_binding_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Darwin build without the kernel-notification binding cannot open local execution."""
    monkeypatch.setattr(local_module.sys, "platform", "darwin")
    monkeypatch.delattr(local_module.select, "kqueue", raising=False)

    with pytest.raises(LocalProcessCleanupError, match="requires Darwin"):
        local_module._require_containment_support()


@_DARWIN_ONLY
def test_snapshot_failure_after_gated_spawn_is_cleaned_without_masking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table failure preserves its error after the blocked root, pipes, and FDs are reaped."""
    marker = tmp_path / "customer-executed"
    runtime = LocalProcessEnvironmentRuntime(
        (sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
        limits=LocalProcessLimits(cleanup_timeout_seconds=0.4),
        workspace_parent=tmp_path,
    )
    monkeypatch.setattr(
        local_module,
        "_process_snapshot",
        lambda: (_ for _ in ()).throw(OSError("injected snapshot failure")),
    )
    before_fds = _open_file_descriptors()
    before_threads = _local_worker_threads()
    started = time.monotonic()
    context = runtime.open(_task())

    with pytest.raises(OSError, match="injected snapshot failure"):
        context.__enter__()

    assert time.monotonic() - started <= 0.65
    assert not marker.exists()
    session = context.__dict__["_session"]
    process = session.__dict__["_process"]
    assert process is not None and process.poll() is not None
    assert _open_file_descriptors() == before_fds
    assert _local_worker_threads() == before_threads
    assert list(tmp_path.glob("exp-sandbox-*")) == []


def _pid_exists(pid: int) -> bool:
    """Return whether one exact process ID still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid_is_running(pid: int) -> bool:
    """Return whether one PID still names a non-zombie process."""
    record = local_module._process_snapshot().get(pid)
    return record is not None and not record.zombie


def _open_file_descriptors() -> frozenset[int]:
    """Return low numbered open FDs without allocating a directory-enumeration descriptor."""
    import fcntl

    open_fds: set[int] = set()
    for file_descriptor in range(256):
        try:
            fcntl.fcntl(file_descriptor, fcntl.F_GETFD)
        except OSError:
            continue
        open_fds.add(file_descriptor)
    return frozenset(open_fds)


def _local_worker_threads() -> frozenset[str]:
    """Return every runtime worker thread name visible to the public test process."""
    return frozenset(
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(("exp-local-descendants-", "exp-local-environment-"))
    )


def _runtime(
    workspace_parent: Path,
    *,
    limits: LocalProcessLimits | None = None,
) -> LocalProcessEnvironmentRuntime:
    """Create one deterministic Python JSONL fixture process without network access."""
    fixture = workspace_parent / "environment_fixture.py"
    fixture.write_text(_FIXTURE_SOURCE, encoding="utf-8")
    return LocalProcessEnvironmentRuntime(
        (sys.executable, str(fixture)),
        limits=limits,
        workspace_parent=workspace_parent,
    )


def _task() -> TaskCase:
    """Return one task whose canonical payload is safe to pass to the fixture process."""
    return TaskCase(
        task_id="task-1",
        lineage_group_id="lineage-1",
        partition="fit",
        instruction="Use the environment fixture.",
        workload_weight=1.0,
        source_trace_ids=("trace-1",),
    )


_FIXTURE_SOURCE = """\
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    if request["kind"] == "open":
        print(json.dumps({"ready": True}), flush=True)
        continue
    if request["kind"] == "close":
        break
    action = request["action"]
    if action["name"] == "sleep":
        time.sleep(1)
    if action["name"] == "crash":
        print("private child stderr", file=sys.stderr, flush=True)
        raise SystemExit(7)
    if action["name"] == "oversized":
        sys.stdout.write("x" * 200)
        sys.stdout.flush()
        continue
    if action["name"] == "malformed":
        print("{not-json", flush=True)
        continue
    if action["name"] == "environment":
        print(json.dumps({"content": "ok", "metadata": {"keys": sorted(os.environ)}}), flush=True)
        continue
    if action["name"] == "orphan-grandchild":
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        print(
            json.dumps(
                {"content": "spawned", "metadata": {"grandchild_pid": child.pid}}
            ),
            flush=True,
        )
        raise SystemExit(0)
    if action["name"] == "detached-grandchild":
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        print(
            json.dumps(
                {"content": "spawned", "metadata": {"grandchild_pid": child.pid}}
            ),
            flush=True,
        )
        raise SystemExit(0)
    if action["name"] == "workspace":
        workspace = Path(os.environ["EXP_SANDBOX_WORKSPACE"])
        (workspace / "tool-output.txt").write_text("written", encoding="utf-8")
        print(json.dumps({"content": "ok", "metadata": {"workspace": str(workspace)}}), flush=True)
        continue
    print(json.dumps({"content": "ok"}), flush=True)
"""

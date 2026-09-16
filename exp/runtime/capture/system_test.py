"""Verify the unprivileged capture supervisor without ever invoking sudo."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from exp.runtime.capture import system
from exp.runtime.capture.system_helper import CaptureSystemError


def test_command_is_isolated_and_exposes_only_validated_parameters() -> None:
    """The elevated interface accepts no environment, filesystem, or executable override."""
    command = system._command("serve", port=18080, domains=("api.openai.com",))
    assert command[:6] == ["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "-S", "-c"]
    assert command[6] == Path(system.__file__).with_name("system_helper.py").read_text()
    assert command[7:] == ["serve", "--port", "18080", "--domain", "api.openai.com"]
    assert system._command("reset")[7:] == ["reset"]


def test_command_snapshots_source_before_later_file_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The elevated command cannot reopen a helper path replaced after construction."""
    source = "import sys; sys.stdout.write('original snapshot')"
    helper = tmp_path / "system_helper.py"
    helper.write_text(source)
    monkeypatch.setattr(system, "__file__", str(tmp_path / "system.py"))
    command = system._command("reset")
    helper.write_text("raise AssertionError('replacement executed')")
    result = subprocess.run(
        [sys.executable, "-I", "-S", *command[5:]], capture_output=True, check=True
    )
    assert result.stdout == b"original snapshot"


def test_root_owned_interpreter_check_rejects_mutable_ancestor(tmp_path: Path) -> None:
    """A world-writable parent cannot become a trusted interpreter location."""
    with pytest.raises(CaptureSystemError, match="root-owned"):
        system._require_root_owned(tmp_path)


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple system Python is a macOS prerequisite")
def test_apple_interpreter_ownership_controls_elevation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept root-controlled installations and reject writable hosted Xcode before sudo."""
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            "import json,sys; sys.stdout.write(json.dumps([sys.executable,sys.base_prefix]))",
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    installed_paths = json.loads(result.stdout)
    assert isinstance(installed_paths, list)
    paths = [Path("/usr/bin/python3")]
    for value in installed_paths:
        assert isinstance(value, str)
        paths.append(Path(value))
    unsafe = False
    for path in paths:
        for candidate in {path, *path.parents, path.resolve(), *path.resolve().parents}:
            info = candidate.stat()
            unsafe |= info.st_uid != 0 or bool(info.st_mode & 0o022)
    if not unsafe:
        system._verify_apple_python()
        return

    real_run = subprocess.run

    def probe_only(
        command: list[str],
        *,
        env: dict[str, str],
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Permit the nonprivileged probe and fail if unsafe Python requests sudo."""
        assert command[0] == "/usr/bin/python3", "unsafe installation requested elevation"
        return real_run(
            command, env=env, capture_output=capture_output, check=check, timeout=timeout
        )

    monkeypatch.setattr(system.subprocess, "run", probe_only)
    with pytest.raises(CaptureSystemError, match="root-owned"):
        system._authorize()


@pytest.mark.parametrize("port, domains", [(443, ("api.openai.com",)), (18080, ("*.openai.com",))])
def test_invalid_setup_does_not_request_admin(
    monkeypatch: pytest.MonkeyPatch, port: int, domains: tuple[str, ...]
) -> None:
    """Unusable settings fail before asking for administrator credentials."""

    def forbidden_authorize() -> None:
        """Fail if the invalid request tries to elevate."""
        pytest.fail("invalid setup requested elevation")

    monkeypatch.setattr(system, "_authorize", forbidden_authorize)
    with pytest.raises(CaptureSystemError):
        system.CaptureSystemSession.start(port, domains)


@pytest.fixture
def fake_helper(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Replace authorization and command construction with a nonprivileged protocol child."""
    script = [
        """
import json, os, sys
assert "EXPLABS_API_KEY" not in os.environ
assert "OPENAI_API_KEY" not in os.environ
sys.stdout.write(json.dumps({"event": "ready"}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    assert line == "PING\\n"
sys.stdout.write(json.dumps({"event": "stopped"}) + "\\n")
sys.stdout.flush()
"""
    ]

    def authorize() -> None:
        """Never invoke sudo or access system authorization in tests."""

    def command(action: str, *, port: int = 0, domains: tuple[str, ...] = ()) -> list[str]:
        """Run only the test script, leaving real process and pipe semantics intact."""
        return [sys.executable, "-I", "-S", "-c", script[0]]

    monkeypatch.setattr(system, "_authorize", authorize)
    monkeypatch.setattr(system, "_command", command)
    monkeypatch.setenv("EXPLABS_API_KEY", "synthetic-secret-not-forwarded")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-secret-not-forwarded")
    yield script


def test_session_start_heartbeat_close_and_secret_isolation(fake_helper: list[str]) -> None:
    """A real child receives heartbeats, observes EOF, and confirms cleanup without secrets."""
    session = system.CaptureSystemSession.start(18080, ("api.openai.com",))
    session.heartbeat()
    session.close(timeout=3)
    assert session.process.returncode == 0
    session.close()
    with pytest.raises(CaptureSystemError, match="helper stopped"):
        session.heartbeat()


def test_start_failure_closes_owner_pipe_and_waits_for_cleanup(fake_helper: list[str]) -> None:
    """A failed startup does not abandon the helper while it might own hosts entries."""
    fake_helper[0] = """
import json, sys
sys.stdout.write(json.dumps({"event": "error", "detail": "synthetic setup failure"}) + "\\n")
sys.stdout.flush()
assert sys.stdin.read() == ""
"""
    with pytest.raises(CaptureSystemError, match="synthetic setup failure"):
        system.CaptureSystemSession.start(18080, ("api.openai.com",))


def test_close_rejects_missing_cleanup_confirmation(fake_helper: list[str]) -> None:
    """Process exit alone is not positive evidence that hosts routing was restored."""
    fake_helper[0] = """
import json, sys
sys.stdout.write(json.dumps({"event": "ready"}) + "\\n")
sys.stdout.flush()
sys.stdin.read()
"""
    session = system.CaptureSystemSession.start(18080, ("api.openai.com",))
    try:
        with pytest.raises(CaptureSystemError, match="before confirming cleanup"):
            session.close(timeout=3)
    finally:
        session.process.wait(timeout=3)


@pytest.mark.parametrize("output", ["not json", "[]", '{"event": 1}', '{"event": "error"}'])
def test_malformed_protocol_is_not_accepted(fake_helper: list[str], output: str) -> None:
    """Malformed and error responses never claim capture readiness."""
    fake_helper[0] = f"import sys; sys.stdout.write({output!r} + '\\n'); sys.stdout.flush()"
    with pytest.raises(CaptureSystemError):
        system.CaptureSystemSession.start(18080, ("api.openai.com",))


def test_reset_is_offline_and_requires_positive_confirmation(fake_helper: list[str]) -> None:
    """Reset needs neither user login nor cloud access and rejects an unsuccessful helper."""
    fake_helper[0] = """
import json, os, sys
assert "EXPLABS_API_KEY" not in os.environ
assert sys.stdin.read() == ""
sys.stdout.write(json.dumps({"event": "reset"}) + "\\n")
"""
    system.reset_capture_system()
    fake_helper[0] = """
import json, sys
sys.stdout.write(json.dumps({"event": "error", "detail": "active capture owns lock"}) + "\\n")
sys.exit(1)
"""
    with pytest.raises(CaptureSystemError, match="active capture owns lock"):
        system.reset_capture_system()


def test_event_reader_timeout_is_bounded() -> None:
    """An unresponsive child cannot block startup forever."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        with pytest.raises(CaptureSystemError, match="timed out"):
            system._read_event(process, 0.05)
    finally:
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=3)
        assert process.stdout is not None
        process.stdout.close()

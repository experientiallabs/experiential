"""Real Unix IPC keeps role capabilities separate and releases its exact resources."""

import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.optimize.evaluation.harbor import client
from exp.optimize.evaluation.harbor.bridge import serve_harbor_trial


class PublicTrial:
    """Finite fake backend; only the transport is under test, never model/native fidelity."""

    def tools(self) -> JsonObject:
        """Expose a single public interface."""
        return {"tools": [{"name": "read"}]}

    def call(self, action: ToolCall) -> JsonObject:
        """Return exact public arguments without any private state."""
        return {"content": json.dumps(action.arguments), "is_error": False}

    def submit(self, answer: JsonObject) -> JsonObject:
        """Acknowledge the actor submission."""
        return {"submitted": True}

    def result(self) -> JsonObject:
        """Only the verifier capability may reach this fixed backend verdict."""
        return {"status": "completed", "reward": 1}

    def output_files(self) -> JsonObject:
        """Expose one explicitly public byte payload."""
        data = b"public-output"
        return {
            "files": [
                {
                    "path": "output.bin",
                    "content_base64": base64.b64encode(data).decode(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            ]
        }


def run_client(
    socket_path: Path, token: str, operation: str, *extra: str, verifier: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run the copied standalone client with only its own role capability and IPC path."""
    environment = {
        "PATH": os.defpath,
        "EXP_HARBOR_SOCKET": str(socket_path),
        "EXP_HARBOR_VERIFIER_TOKEN" if verifier else "EXP_HARBOR_ACTOR_TOKEN": token,
    }
    return subprocess.run(
        [sys.executable, str(Path(client.__file__)), operation, *extra],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_unix_standalone_roundtrip_and_private_grade_capability(tmp_path: Path) -> None:
    """Actual IPC handles public calls/output and denies actor access to private grading."""
    directory = tmp_path / "ipc"
    directory.mkdir(mode=0o755)
    path = directory / "backend.sock"
    with serve_harbor_trial(PublicTrial(), socket_path=path) as endpoint:
        assert endpoint.socket_path == path
        assert json.loads(run_client(path, endpoint.actor_token, "tools").stdout) == {
            "tools": [{"name": "read"}]
        }
        body = json.dumps({"call_id": "c1", "name": "read", "arguments": {"x": 1}})
        call = run_client(path, endpoint.actor_token, "call", "--body", body)
        assert call.returncode == 0 and json.loads(call.stdout)["content"] == '{"x": 1}'
        assert (
            run_client(
                path, endpoint.actor_token, "submit", "--body", '{"answer":"done"}'
            ).returncode
            == 0
        )
        denied = run_client(
            path,
            endpoint.actor_token,
            "verify",
            "--directory",
            str(tmp_path / "denied"),
            verifier=True,
        )
        assert (
            denied.returncode == 2
            and '"http_status": 403' in (tmp_path / "denied/experiential.json").read_text()
        )
        assert not (tmp_path / "denied/reward.json").exists()
        graded = run_client(
            path,
            endpoint.verifier_token,
            "verify",
            "--directory",
            str(tmp_path / "grade"),
            verifier=True,
        )
        assert graded.returncode == 0
        assert json.loads((tmp_path / "grade/reward.json").read_text()) == {
            "reward": 1,
            "scored": 1,
        }
        output = run_client(
            path, endpoint.actor_token, "outputs", "--directory", str(tmp_path / "outputs")
        )
        assert (
            output.returncode == 0
            and (tmp_path / "outputs/output.bin").read_bytes() == b"public-output"
        )
        assert list(directory.iterdir()) == [path]
    assert not path.exists() and list(directory.iterdir()) == []


@pytest.mark.parametrize("kind", ["existing", "symlink", "private_file", "tcp_override"])
def test_unix_startup_refuses_unowned_or_mixed_socket_directory(tmp_path: Path, kind: str) -> None:
    """A socket mount cannot include credentials, reuse a listener or follow links."""
    directory = tmp_path / "ipc"
    directory.mkdir(mode=0o755)
    path = directory / "backend.sock"
    if kind == "existing":
        path.write_text("do not remove")
    elif kind == "symlink":
        path.symlink_to(tmp_path / "absent")
    elif kind == "private_file":
        (directory / "capabilities.json").write_text("private")
    with pytest.raises(ValueError):
        with serve_harbor_trial(
            PublicTrial(), socket_path=path, port=123 if kind == "tcp_override" else 0
        ):
            pytest.fail("invalid startup unexpectedly succeeded")
    if kind == "existing":
        assert path.read_text() == "do not remove"
    if kind == "symlink":
        assert path.is_symlink()


def test_unix_cleanup_never_unlinks_replacement_inode(tmp_path: Path) -> None:
    """Cleanup owns the created socket inode, not every later occupant of its pathname."""
    directory = tmp_path / "ipc"
    directory.mkdir(mode=0o755)
    path = directory / "backend.sock"
    with serve_harbor_trial(PublicTrial(), socket_path=path):
        path.unlink()
        path.write_text("replacement")
    assert path.read_text() == "replacement"


def test_repeated_unix_open_close_drains_partial_reads_without_fd_or_thread_growth(
    tmp_path: Path,
) -> None:
    """A stalled incomplete HTTP header is interrupted on close rather than leaking a worker."""
    directory = tmp_path / "ipc"
    directory.mkdir(mode=0o755)
    path = directory / "backend.sock"
    fd_before = len(tuple(Path("/proc/self/fd").iterdir()))
    threads_before = set(threading.enumerate())
    started = time.monotonic()
    for _ in range(5):
        with serve_harbor_trial(PublicTrial(), socket_path=path):
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(str(path))
            connection.sendall(b"POST /call HTTP/1.1\r\nContent-Length: 100\r\n")
            time.sleep(0.02)
        connection.close()
        assert not path.exists()
        assert len(tuple(Path("/proc/self/fd").iterdir())) <= fd_before + 1
        assert not set(threading.enumerate()) - threads_before
    assert time.monotonic() - started < 10


def test_unix_partial_startup_failure_closes_fd_and_owned_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure after bind but before listening cannot leave an owned socket or descriptor."""
    from exp.optimize.evaluation.harbor.bridge import _UnixServer

    directory = tmp_path / "ipc"
    directory.mkdir(mode=0o755)
    path = directory / "backend.sock"
    before = len(tuple(Path("/proc/self/fd").iterdir()))

    def fail_activation(self: _UnixServer) -> None:
        """Refuse activation after the real bind created its socket inode."""
        raise RuntimeError("activation failed")

    monkeypatch.setattr(_UnixServer, "server_activate", fail_activation)
    with pytest.raises(RuntimeError, match="activation failed"):
        with serve_harbor_trial(PublicTrial(), socket_path=path):
            pytest.fail("startup must fail")
    assert not path.exists()
    assert len(tuple(Path("/proc/self/fd").iterdir())) <= before

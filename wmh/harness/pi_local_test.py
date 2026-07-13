"""Local pi runner tests: runtime bootstrap and stdio frame transport."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import cast

import pytest

import wmh.harness.pi_local as mod
from wmh.harness.pi_local import (
    LocalStdioChannel,
    ensure_local_pi_runtime,
    parse_node_version,
)


class _FakeResult:
    """Completed-command slice returned by bootstrap test doubles."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_parse_node_version_requires_semver_shape() -> None:
    """Node's normal version output parses; unrelated output fails loudly."""
    assert parse_node_version("v22.19.0\n") == (22, 19, 0)
    with pytest.raises(RuntimeError, match="could not parse"):
        parse_node_version("node unknown")


def test_runtime_bootstrap_installs_once(tmp_path: Path) -> None:
    """The runner refreshes every time while pinned npm dependencies install once."""
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> _FakeResult:
        calls.append(command)
        if command[-1] == "--version":
            return _FakeResult("v22.19.0\n")
        return _FakeResult("")

    runtime = ensure_local_pi_runtime(
        tmp_path,
        node="node",
        npm="npm",
        run_command=run,
    )
    assert runtime == tmp_path
    assert (tmp_path / "runner_live.ts").is_file()
    assert (tmp_path / "package.json").is_file()
    assert any(command[:2] == ["npm", "install"] for command in calls)

    calls.clear()
    ensure_local_pi_runtime(tmp_path, node="node", npm="npm", run_command=run)
    assert calls == [["node", "--version"]]


def test_runtime_bootstrap_rejects_old_node(tmp_path: Path) -> None:
    """The local harness fails before npm work when Node cannot strip pi's TypeScript."""

    def run(_command: list[str], **_kwargs: object) -> _FakeResult:
        return _FakeResult("v20.10.0\n")

    with pytest.raises(RuntimeError, match="Node.js 22.19"):
        ensure_local_pi_runtime(tmp_path, node="node", npm="npm", run_command=run)


class _FakeProcess:
    """Minimal text-mode Popen stand-in for LocalStdioChannel."""

    def __init__(self, frames: list[dict[str, object]]) -> None:
        encoded = "".join(
            base64.b64encode(json.dumps(frame).encode()).decode() + "\n" for frame in frames
        )
        self.stdout = io.StringIO(encoded)
        self.stderr = io.StringIO("")
        self.stdin = io.StringIO()
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        self.terminated = True
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


def test_local_stdio_channel_round_trips_frames_and_cleans_run_dir(tmp_path: Path) -> None:
    """The channel transports frames, closes its process, and removes the private cwd."""
    process = _FakeProcess([{"type": "hello", "mode": "session"}])
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    channel = LocalStdioChannel(cast("mod._TextProcess", process), cleanup_dir=run_dir)

    assert channel.recv(timeout=1) == {"type": "hello", "mode": "session"}
    channel.send({"type": "ping", "nonce": "n1"})

    wire = process.stdin.getvalue().strip()
    assert json.loads(base64.b64decode(wire)) == {"type": "ping", "nonce": "n1"}
    channel.close()
    assert process.terminated
    assert not run_dir.exists()

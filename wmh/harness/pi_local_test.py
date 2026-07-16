"""Local pi runner tests: runtime bootstrap and stdio frame transport."""

from __future__ import annotations

import base64
import io
import json
import os
import select
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

import wmh.harness.pi_local as mod
from wmh.harness.pi_e2b import TRANSPORT_KEEPALIVE_TYPE
from wmh.harness.pi_local import (
    LocalStdioChannel,
    ensure_local_pi_runtime,
    parse_node_version,
)


class _FakeResult:
    """Completed-command slice returned by bootstrap test doubles."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def _node_22() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run(  # noqa: S603 - resolved local Node executable, no shell
        [node, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    if parse_node_version(result.stdout) < (22, 19, 0):
        pytest.skip("runner_live.ts requires Node.js 22.19+")
    return node


def _start_live_runner(
    node: str, env: dict[str, str], *, cwd: Path | None = None
) -> subprocess.Popen[str]:
    runner = Path(mod.__file__).with_name("pi_entry") / "runner_live.ts"
    return subprocess.Popen(  # noqa: S603 - resolved local Node executable, no shell
        [node, "--experimental-strip-types", str(runner)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd,
    )


def _read_live_frame(process: subprocess.Popen[str]) -> dict[str, object]:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 5)
    assert ready, "live runner did not emit a frame within five seconds"
    wire = process.stdout.readline().strip()
    if not wire:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise AssertionError(f"live runner exited before its next frame: {stderr}")
    return cast("dict[str, object]", json.loads(base64.b64decode(wire)))


def _send_live_frame(process: subprocess.Popen[str], frame: dict[str, object]) -> None:
    assert process.stdin is not None
    wire = base64.b64encode(json.dumps(frame).encode()).decode() + "\n"
    process.stdin.write(wire)
    process.stdin.flush()


def _stop_live_runner(
    process: subprocess.Popen[str], *, durable_inbound_seq: int | None = None
) -> None:
    if process.poll() is None:
        try:
            frame: dict[str, object] = {"type": "shutdown"}
            if durable_inbound_seq is not None:
                frame = {"transport_in_seq": durable_inbound_seq, "frame": frame}
            _send_live_frame(process, frame)
            process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


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


def test_live_runner_default_output_keeps_the_legacy_frame_shape() -> None:
    """Without an outbox opt-in, stdout remains the original unwrapped frame stream."""
    node = _node_22()
    env = os.environ.copy()
    env.pop("WMH_LIVE_OUTBOX", None)
    env["NODE_NO_WARNINGS"] = "1"
    process = _start_live_runner(node, env)
    try:
        hello = _read_live_frame(process)
        assert hello["type"] == "hello"
        assert "transport_seq" not in hello
        assert "frame" not in hello
    finally:
        _stop_live_runner(process)


def test_live_runner_durable_outbox_precedes_sequenced_stdout(tmp_path: Path) -> None:
    """Semantic frames are committed in sequence before their matching envelopes reach stdout."""
    node = _node_22()
    outbox = tmp_path / "live-outbox"
    env = os.environ.copy()
    env["NODE_NO_WARNINGS"] = "1"
    env["WMH_LIVE_OUTBOX"] = str(outbox)
    process = _start_live_runner(node, env)
    try:
        hello = _read_live_frame(process)
        assert hello["transport_seq"] == 1
        assert cast("dict[str, object]", hello["frame"])["type"] == "hello"
        # Seeing stdout is sufficient proof that both atomic outbox writes have completed: publish
        # is synchronous and send writes stdout only after publishing the frame and head.
        assert (outbox / "head").read_text() == "1\n"
        assert json.loads((outbox / "frames" / "00000000000000000001.json").read_text()) == hello

        inbound: dict[str, object] = {
            "transport_in_seq": 1,
            "frame": {"type": "ping", "nonce": "n1"},
        }
        _send_live_frame(process, inbound)
        pong = _read_live_frame(process)
        assert pong == {
            "transport_seq": 2,
            "frame": {"type": "pong", "nonce": "n1"},
        }
        ack = _read_live_frame(process)
        assert ack == {
            "transport_seq": 3,
            "frame": {"type": "transport_ack", "transport_in_seq": 1},
        }

        # A physical resend after an ambiguous HTTP timeout repairs the lost ack but must not
        # dispatch the logical ping twice.
        _send_live_frame(process, inbound)
        duplicate_ack = _read_live_frame(process)
        assert duplicate_ack == {
            "transport_seq": 4,
            "frame": {"type": "transport_ack", "transport_in_seq": 1},
        }
        assert (outbox / "head").read_text() == "4\n"
        assert json.loads((outbox / "frames" / "00000000000000000002.json").read_text()) == pong
        assert json.loads((outbox / "frames" / "00000000000000000003.json").read_text()) == ack
        assert (
            json.loads((outbox / "frames" / "00000000000000000004.json").read_text())
            == duplicate_ack
        )
        assert list(outbox.rglob(".*.tmp-*")) == []
    finally:
        _stop_live_runner(process, durable_inbound_seq=2)


def test_live_runner_turn_scope_clears_only_the_prior_outer_turn(tmp_path: Path) -> None:
    """Project turns reuse one runner but never accumulate an unbounded chat transcript."""
    node = _node_22()
    env = os.environ.copy()
    env.pop("WMH_LIVE_OUTBOX", None)
    env["NODE_NO_WARNINGS"] = "1"
    process = _start_live_runner(node, env, cwd=tmp_path)
    agent_source = """export class Agent {
  state = { messages: [] };
  listeners = [];
  constructor(_options) {}
  subscribe(listener) { this.listeners.push(listener); return () => {}; }
  steer(_message) {}
  abort() {}
  async prompt(text) {
    if (this.state.messages.length !== 0) throw new Error("prior outer turn was retained");
    this.state.messages = [{ role: "user", text }];
    for (const listener of this.listeners) await listener({ type: "turn_end" });
  }
}
"""
    try:
        assert _read_live_frame(process)["type"] == "hello"
        _send_live_frame(
            process,
            {
                "type": "session_start",
                "files": {"src/agent.ts": agent_source},
                "tools": [],
                "conversation_scope": "turn",
            },
        )
        assert _read_live_frame(process) == {"type": "state", "status": "idle", "turns": 0}

        for index in range(2):
            _send_live_frame(
                process,
                {"type": "user_message", "msg_id": f"m{index}", "text": f"round {index}"},
            )
            assert _read_live_frame(process)["status"] == "running"
            terminal = _read_live_frame(process)
            assert terminal["status"] == "idle"
            assert terminal["reason"] == "completed"
    finally:
        _stop_live_runner(process)


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


def test_local_stdio_channel_filters_transport_keepalives() -> None:
    """Runner liveness is transport-only and never leaks into an interactive session."""
    process = _FakeProcess(
        [
            {"type": TRANSPORT_KEEPALIVE_TYPE},
            {"type": "hello", "mode": "session"},
        ]
    )
    channel = LocalStdioChannel(cast("mod._TextProcess", process))

    assert channel.recv(timeout=1) == {"type": "hello", "mode": "session"}
    channel.close()

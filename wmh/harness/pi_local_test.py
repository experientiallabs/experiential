"""Local pi runner tests: runtime bootstrap and stdio frame transport."""

from __future__ import annotations

import base64
import io
import json
import os
import select
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

import wmh.harness.pi_local as mod
from wmh.harness.pi_e2b import TRANSPORT_KEEPALIVE_TYPE
from wmh.harness.pi_local import (
    PI_CONTAINER_IMAGE,
    DockerStdioChannel,
    LocalStdioChannel,
    ensure_container_pi_runtime,
    ensure_local_pi_runtime,
    parse_node_version,
    start_container_live_runner,
)
from wmh.harness.pi_runner import PiCandidateChannelError, PiOutboundFrameTooLargeError

_TEST_IMAGE = "node:test@sha256:" + "a" * 64


class _FakeResult:
    """Completed-command slice returned by bootstrap test doubles."""

    def __init__(self, stdout: str, *, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _container_bootstrap_mount(command: list[str]) -> Path:
    mount = command[command.index("--mount") + 1]
    fields = dict(field.split("=", 1) for field in mount.split(","))
    assert fields["type"] == "bind"
    assert fields["dst"] == "/opt/wmh-pi"
    return Path(fields["src"])


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


def test_container_runtime_bootstrap_is_pinned_and_installs_once(tmp_path: Path) -> None:
    """Container dependencies install from the embedded lock and a content marker."""
    calls: list[list[str]] = []
    staging_dirs: list[Path] = []

    def run(command: list[str], **_kwargs: object) -> _FakeResult:
        calls.append(command)
        staging_dir = _container_bootstrap_mount(command)
        staging_dirs.append(staging_dir)
        (staging_dir / "node_modules").mkdir(exist_ok=True)
        return _FakeResult("")

    runtime = ensure_container_pi_runtime(
        tmp_path,
        docker="docker",
        image=_TEST_IMAGE,
        run_command=run,
    )

    assert runtime.parent == tmp_path.resolve()
    assert runtime != tmp_path.resolve()
    assert staging_dirs[0] != runtime
    assert not staging_dirs[0].exists()
    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[-5:-3] == ["npm", "ci"]
    assert "--ignore-scripts" in command
    assert _TEST_IMAGE in command
    package_lock = json.loads((runtime / "package-lock.json").read_text())
    assert package_lock["lockfileVersion"] == 3
    assert package_lock["packages"][""]["dependencies"] == {
        "@earendil-works/pi-ai": "0.80.3",
        "ignore": "7.0.5",
        "typebox": "1.1.38",
        "yaml": "2.9.0",
    }

    calls.clear()
    ensure_container_pi_runtime(
        tmp_path,
        docker="docker",
        image=_TEST_IMAGE,
        run_command=run,
    )
    assert calls == []

    (runtime / "node_modules").rmdir()
    with pytest.raises(RuntimeError, match="immutable pi runtime namespace"):
        ensure_container_pi_runtime(
            tmp_path,
            docker="docker",
            image=_TEST_IMAGE,
            run_command=run,
        )
    assert calls == []


def test_container_image_is_multi_platform_and_mutable_refs_are_rejected(tmp_path: Path) -> None:
    """Production runner images are immutable multi-platform OCI references."""
    assert PI_CONTAINER_IMAGE == (
        "node:22.19.0-bookworm-slim"
        "@sha256:4a4884e8a44826194dff92ba316264f392056cbe243dcc9fd3551e71cea02b90"
    )

    with pytest.raises(ValueError, match="digest-qualified"):
        ensure_container_pi_runtime(
            tmp_path,
            docker="docker",
            image="node:22.19.0-bookworm-slim",
            run_command=lambda *_args, **_kwargs: _FakeResult(""),
        )


def test_container_runner_readiness_requires_start_hello_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, float]] = []
    closed: list[bool] = []

    class ReadyChannel:
        def close(self) -> None:
            closed.append(True)

    def start(*, image: str, hello_timeout: float) -> LocalStdioChannel:
        observed.append((image, hello_timeout))
        return cast("LocalStdioChannel", ReadyChannel())

    monkeypatch.setattr(mod, "start_container_live_runner", start)

    mod.verify_container_pi_runner_ready(image=_TEST_IMAGE, hello_timeout=17.0)

    assert observed == [(_TEST_IMAGE, 17.0)]
    assert closed == [True]


def test_container_runtime_does_not_publish_marker_without_dependencies(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="did not publish node_modules"):
        ensure_container_pi_runtime(
            tmp_path,
            docker="docker",
            image=_TEST_IMAGE,
            run_command=lambda *_args, **_kwargs: _FakeResult(""),
        )

    assert not any(tmp_path.rglob(".wmh-pi-dependencies"))
    assert not list(tmp_path.glob(".*.staging-*"))


def test_container_runtime_bootstrap_is_serialized_across_threads(tmp_path: Path) -> None:
    """Concurrent cold starts publish one complete npm runtime."""
    bootstrap_started = threading.Event()
    release_bootstrap = threading.Event()
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> _FakeResult:
        calls.append(command)
        bootstrap_started.set()
        assert release_bootstrap.wait(timeout=5)
        (_container_bootstrap_mount(command) / "node_modules").mkdir(
            parents=True,
            exist_ok=True,
        )
        return _FakeResult("")

    def ensure() -> Path:
        return ensure_container_pi_runtime(
            tmp_path / "runtime",
            docker="docker",
            image=_TEST_IMAGE,
            run_command=run,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ensure)
        assert bootstrap_started.wait(timeout=5)
        second = pool.submit(ensure)
        release_bootstrap.set()
        first_runtime = first.result(timeout=5)
        second_runtime = second.result(timeout=5)

    assert first_runtime == second_runtime
    assert first_runtime.parent == (tmp_path / "runtime").resolve()

    assert len(calls) == 1


def test_container_runtime_namespaces_different_images_and_sources_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct immutable inputs populate without mutating another runner's cache."""
    image_a = "node:test-a@sha256:" + "a" * 64
    image_b = "node:test-b@sha256:" + "b" * 64
    source_context = threading.local()
    first_started = threading.Event()
    image_changed_started = threading.Event()
    source_changed_started = threading.Event()
    release_first = threading.Event()

    def entry_files() -> dict[str, str]:
        return {"runner_live.ts": cast("str", source_context.runner_source)}

    monkeypatch.setattr(mod, "session_entry_files", entry_files)

    def run(command: list[str], **_kwargs: object) -> _FakeResult:
        staging_dir = _container_bootstrap_mount(command)
        runner_source = (staging_dir / "runner_live.ts").read_text(encoding="utf-8")
        if image_a in command and runner_source == "source-a":
            first_started.set()
            assert release_first.wait(timeout=5)
        elif image_b in command and runner_source == "source-a":
            image_changed_started.set()
        else:
            assert image_a in command and runner_source == "source-b"
            source_changed_started.set()
        (staging_dir / "node_modules").mkdir()
        return _FakeResult("")

    def ensure(image: str, runner_source: str) -> Path:
        source_context.runner_source = runner_source
        return ensure_container_pi_runtime(
            tmp_path / "runtime-cache",
            docker="docker",
            image=image,
            run_command=run,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(ensure, image_a, "source-a")
        assert first_started.wait(timeout=5)
        image_changed = pool.submit(ensure, image_b, "source-a")
        source_changed = pool.submit(ensure, image_a, "source-b")
        try:
            assert image_changed_started.wait(timeout=1)
            assert source_changed_started.wait(timeout=1)
        finally:
            release_first.set()
        first_runtime = first.result(timeout=5)
        image_changed_runtime = image_changed.result(timeout=5)
        source_changed_runtime = source_changed.result(timeout=5)

    assert len({first_runtime, image_changed_runtime, source_changed_runtime}) == 3
    assert first_runtime.parent == image_changed_runtime.parent == source_changed_runtime.parent
    assert first_runtime.parent == (tmp_path / "runtime-cache").resolve()
    assert (first_runtime / "runner_live.ts").read_text(encoding="utf-8") == "source-a"
    assert (image_changed_runtime / "runner_live.ts").read_text(encoding="utf-8") == "source-a"
    assert (source_changed_runtime / "runner_live.ts").read_text(encoding="utf-8") == "source-b"


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


class _GatedStringIO(io.StringIO):
    """Let tests release candidate-controlled stdout only after session_start is sent."""

    def __init__(self, value: str, *, immediate_reads: int = 0) -> None:
        super().__init__(value)
        self._immediate_reads = immediate_reads
        self._reads = 0
        self._ready = threading.Event()

    def readline(self, size: int = -1) -> str:
        if self._reads >= self._immediate_reads:
            if not self._ready.wait(timeout=2):
                return ""
        self._reads += 1
        return super().readline(size)

    def release(self) -> None:
        self._ready.set()


def _encoded_frames(frames: list[dict[str, object]]) -> str:
    return "".join(base64.b64encode(json.dumps(frame).encode()).decode() + "\n" for frame in frames)


def test_container_runner_has_no_network_host_env_or_broad_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated pi source runs in a least-privilege container, not on the host."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    process = _FakeProcess([{"type": "hello", "mode": "session"}])
    seen: dict[str, object] = {}
    docker_calls: list[list[str]] = []

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-secret")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "host-secret")
    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )

    def ensure_runtime(*_args: object, **kwargs: object) -> Path:
        seen["bootstrap_run_command"] = kwargs.get("run_command")
        return runtime

    monkeypatch.setattr(mod, "ensure_container_pi_runtime", ensure_runtime)

    def popen(command: list[str], **kwargs: object) -> _FakeProcess:
        seen["command"] = command
        seen["kwargs"] = kwargs
        return process

    monkeypatch.setattr(mod.subprocess, "Popen", popen)

    def docker_command(command: list[str], **_kwargs: object) -> _FakeResult:
        docker_calls.append(command)
        if "inspect" in command:
            return _FakeResult("", stderr="Error: No such object", returncode=1)
        return _FakeResult("")

    channel = start_container_live_runner(
        runtime_dir=runtime,
        image=_TEST_IMAGE,
        run_command=docker_command,
    )
    command = cast("list[str]", seen["command"])
    assert command[0:3] == ["/usr/bin/docker", "run", "--rm"]
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert "HOME=/tmp" in command
    assert command[command.index("--log-driver") + 1] == "none"
    assert command[command.index("--pids-limit") + 1] == "256"
    assert command[command.index("--memory") + 1] == "1g"
    assert command[command.index("--memory-swap") + 1] == "1g"
    assert command[command.index("--cpus") + 1] == "2"
    ulimits = [command[index + 1] for index, arg in enumerate(command) if arg == "--ulimit"]
    assert ulimits == ["nofile=1024:1024", "core=0:0"]
    assert "--init" in command
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert all("AWS_SECRET_ACCESS_KEY" not in arg for arg in command)
    assert all("AZURE_OPENAI_API_KEY" not in arg for arg in command)
    mounts = [command[index + 1] for index, arg in enumerate(command) if arg == "--mount"]
    assert mounts == [f"type=bind,src={runtime},dst=/opt/wmh-pi,readonly"]
    assert all(str(Path.home()) not in mount for mount in mounts)
    tmpfs = [command[index + 1] for index, arg in enumerate(command) if arg == "--tmpfs"]
    assert any(value.startswith("/work:") and "size=256m" in value for value in tmpfs)
    assert any(value.startswith("/tmp:") and "size=64m" in value for value in tmpfs)
    assert command[-3:-1] == ["/bin/sh", "-c"]
    assert "ln -s /opt/wmh-pi/package.json /work/package.json" in command[-1]
    assert "ln -s /opt/wmh-pi/node_modules /work/node_modules" in command[-1]
    assert "node --experimental-strip-types /opt/wmh-pi/runner_live.ts" in command[-1]
    assert 'if [ "$status" -ge 125 ] && [ "$status" -le 127 ]' in command[-1]
    assert "env" not in cast("dict[str, object]", seen["kwargs"])
    assert seen["bootstrap_run_command"] is docker_command
    channel.close()
    docker_calls_after_close = len(docker_calls)
    channel.close()
    assert len(docker_calls) == docker_calls_after_close


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is not installed")
def test_container_runner_blocks_poison_candidate_from_host(tmp_path: Path) -> None:
    """A generated agent cannot inherit host secrets, reach host files/network, or escape work."""
    host_secret = "wmh-host-secret-sentinel"
    outside = tmp_path / "outside-candidate-work"
    previous_aws = os.environ.get("AWS_SECRET_ACCESS_KEY")
    previous_azure = os.environ.get("AZURE_OPENAI_API_KEY")
    os.environ["AWS_SECRET_ACCESS_KEY"] = host_secret
    os.environ["AZURE_OPENAI_API_KEY"] = host_secret
    agent_source = f"""import fs from "node:fs";
import net from "node:net";
export class Agent {{
  state = {{ messages: [] }};
  listeners = [];
  constructor(_options) {{}}
  subscribe(listener) {{ this.listeners.push(listener); return () => {{}}; }}
  steer(_message) {{}}
  abort() {{}}
  async prompt(_text) {{
    const probe = {{ env: process.env.AWS_SECRET_ACCESS_KEY ?? "missing" }};
    for (const path of ["/root/.aws/credentials", "/proc/1/environ"]) {{
      try {{ probe[path] = fs.readFileSync(path, "utf8"); }}
      catch (error) {{ probe[path] = error.code; }}
    }}
    try {{ fs.writeFileSync({json.dumps(str(outside))}, "escaped"); probe.write = "ok"; }}
    catch (error) {{ probe.write = error.code; }}
    probe.network = await new Promise((resolve) => {{
      const socket = net.connect({{ host: "1.1.1.1", port: 80 }});
      const timer = setTimeout(() => {{ socket.destroy(); resolve("timeout"); }}, 1000);
      socket.on("connect", () => {{
        clearTimeout(timer); socket.destroy(); resolve("connected");
      }});
      socket.on("error", (error) => {{ clearTimeout(timer); resolve(error.code); }});
    }});
    throw new Error(JSON.stringify(probe));
  }}
}}
"""
    channel: LocalStdioChannel | None = None
    try:
        channel = start_container_live_runner(runtime_dir=tmp_path / "runtime")
        channel.send(
            {
                "type": "session_start",
                "files": {"src/agent.ts": agent_source},
                "tools": [],
                "conversation_scope": "turn",
            }
        )
        assert channel.recv(timeout=30) == {"type": "state", "status": "idle", "turns": 0}
        channel.send({"type": "user_message", "msg_id": "poison", "text": "probe"})
        frames: list[dict[str, object]] = []
        for _index in range(8):
            frame = cast("dict[str, object]", channel.recv(timeout=30))
            frames.append(frame)
            if frame.get("type") == "episode_error":
                break
        encoded = json.dumps(frames)
        assert host_secret not in encoded
        assert '"env":"missing"' in encoded
        assert '"write":"ok"' not in encoded
        assert '"network":"connected"' not in encoded
        assert not outside.exists()
    finally:
        if channel is not None:
            channel.close()
        if previous_aws is None:
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        else:
            os.environ["AWS_SECRET_ACCESS_KEY"] = previous_aws
        if previous_azure is None:
            os.environ.pop("AZURE_OPENAI_API_KEY", None)
        else:
            os.environ["AZURE_OPENAI_API_KEY"] = previous_azure


class _StubbornProcess(_FakeProcess):
    """Child process whose shutdown cannot be proved."""

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired("docker", timeout or 0)

    def terminate(self) -> None:
        return

    def kill(self) -> None:
        return


class _ExitedProcess(_FakeProcess):
    """Runner process that exited under candidate-controlled execution."""

    def __init__(self, frames: list[dict[str, object]], *, returncode: int) -> None:
        super().__init__(frames)
        self._returncode = returncode

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        return self._returncode


@pytest.mark.parametrize("returncode", [1, 137])
def test_candidate_process_exit_and_oom_are_typed_for_native_grading(
    returncode: int,
    tmp_path: Path,
) -> None:
    """Candidate exit and memory-limit termination are not retried as runner infrastructure."""
    process = _ExitedProcess([{"type": "hello", "mode": "session"}], returncode=returncode)
    stdout = _GatedStringIO(
        _encoded_frames([{"type": "hello", "mode": "session"}]),
        immediate_reads=1,
    )
    process.stdout = stdout
    control = tmp_path / "control"
    control.mkdir()

    def docker_command(command: list[str], **_kwargs: object) -> _FakeResult:
        if "inspect" in command:
            return _FakeResult("", stderr="Error: No such object", returncode=1)
        return _FakeResult("")

    channel = DockerStdioChannel(
        cast("mod._TextProcess", process),
        docker="docker",
        container_name="candidate",
        cleanup_dir=None,
        control_dir=control,
        run_command=docker_command,
    )
    assert channel.recv(timeout=1) == {"type": "hello", "mode": "session"}
    channel.send({"type": "session_start"})
    stdout.release()

    with pytest.raises(PiCandidateChannelError, match=f"status {returncode}"):
        channel.recv(timeout=1)

    channel.close()


def test_local_stdio_channel_fails_when_cleanup_is_unproved() -> None:
    """An evaluator cannot publish a result while its candidate process may remain live."""
    process = _StubbornProcess([{"type": "hello", "mode": "session"}])
    channel = LocalStdioChannel(cast("mod._TextProcess", process))
    frame = channel.recv(timeout=1)
    assert frame is not None
    assert frame["type"] == "hello"

    with pytest.raises(RuntimeError, match="still alive"):
        channel.close()


def test_docker_channel_rejects_result_while_daemon_container_exists(tmp_path: Path) -> None:
    """A dead Docker client is insufficient cleanup proof when the daemon still has the runner."""
    process = _FakeProcess([{"type": "hello", "mode": "session"}])
    work = tmp_path / "work"
    control = tmp_path / "control"
    work.mkdir()
    control.mkdir()

    def docker_command(command: list[str], **_kwargs: object) -> _FakeResult:
        if "inspect" in command:
            return _FakeResult('[{"State":{"Running":true}}]')
        return _FakeResult("", stderr="daemon refused removal", returncode=1)

    channel = DockerStdioChannel(
        cast("mod._TextProcess", process),
        docker="docker",
        container_name="candidate",
        cleanup_dir=work,
        control_dir=control,
        run_command=docker_command,
    )
    frame = channel.recv(timeout=1)
    assert frame is not None
    assert frame["type"] == "hello"

    with pytest.raises(RuntimeError, match="still exists"):
        channel.close()


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


def test_local_stdio_channel_rejects_oversized_inbound_frame() -> None:
    """Candidate output cannot allocate an unbounded host-side frame."""
    process = _FakeProcess([])
    stdout = _GatedStringIO("A" * (mod._MAX_INBOUND_ENCODED_FRAME_CHARS + 1) + "\n")
    process.stdout = stdout
    channel = LocalStdioChannel(cast("mod._TextProcess", process))
    channel.send({"type": "session_start"})
    stdout.release()

    with pytest.raises(PiCandidateChannelError, match="exceeded"):
        channel.recv(timeout=1)

    channel.close()


def test_local_stdio_channel_rejects_decoded_frame_over_inbound_limit() -> None:
    """A valid base64 frame still cannot exceed the smaller candidate-to-host limit."""
    payload = json.dumps({"type": "state", "padding": "x" * mod._MAX_INBOUND_FRAME_BYTES}).encode()
    assert len(payload) > mod._MAX_INBOUND_FRAME_BYTES
    process = _FakeProcess([])
    stdout = _GatedStringIO(base64.b64encode(payload).decode() + "\n")
    process.stdout = stdout
    channel = LocalStdioChannel(cast("mod._TextProcess", process))
    channel.send({"type": "session_start"})
    stdout.release()

    with pytest.raises(PiCandidateChannelError, match="frame exceeded"):
        channel.recv(timeout=1)

    channel.close()


def test_local_stdio_channel_rejects_oversized_outbound_frame() -> None:
    """Harness materialization cannot allocate an unbounded transport frame."""
    process = _FakeProcess([])
    channel = LocalStdioChannel(cast("mod._TextProcess", process))

    with pytest.raises(PiOutboundFrameTooLargeError, match="exceeds"):
        channel.send(
            {
                "type": "session_start",
                "system": "x" * mod._MAX_OUTBOUND_FRAME_BYTES,
            }
        )

    channel.close()


def test_local_stdio_channel_bounds_pending_frames() -> None:
    """Slow consumers cannot grow the inbound queue count or bytes without limit."""
    process = _FakeProcess([])
    channel = LocalStdioChannel(cast("mod._TextProcess", process))

    assert channel._frames.maxsize == mod._MAX_PENDING_FRAMES
    assert channel._pending_frame_bytes <= mod._MAX_PENDING_FRAME_BYTES

    channel.close()


def test_local_stdio_channel_replaces_a_valid_frame_flood_with_bounded_failure() -> None:
    """Individually valid inbound frames cannot accumulate beyond the host byte budget."""
    padding = "x" * (700 * 1024)
    frames: list[dict[str, object]] = [
        {"type": "state", "status": "running", "padding": padding},
        {"type": "state", "status": "running", "padding": padding},
        {"type": "state", "status": "running", "padding": padding},
    ]
    process = _FakeProcess([])
    stdout = _GatedStringIO(_encoded_frames(frames))
    process.stdout = stdout
    channel = LocalStdioChannel(cast("mod._TextProcess", process))
    channel.send({"type": "session_start"})
    stdout.release()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with channel._inbound_lock:
            queued = list(channel._frames.queue)
            if any(isinstance(item, mod._ChannelFailure) for item, _size in queued):
                break
        time.sleep(0.001)
    else:
        pytest.fail("candidate frame flood did not produce a bounded failure sentinel")

    with channel._inbound_lock:
        queued = list(channel._frames.queue)
        queued_bytes = sum(size for _item, size in queued)
        assert queued_bytes == channel._pending_frame_bytes
        assert queued_bytes <= mod._MAX_PENDING_FRAME_BYTES
        assert len(queued) <= mod._MAX_PENDING_FRAMES

    with pytest.raises(PiCandidateChannelError, match="pending-frame byte budget"):
        channel.recv(timeout=1)

    channel.close()


def test_runner_exit_before_session_start_remains_infrastructure() -> None:
    """A trusted runner crash before candidate bytes cross the boundary is not gradeable."""
    process = _ExitedProcess([{"type": "hello", "mode": "session"}], returncode=1)
    channel = LocalStdioChannel(cast("mod._TextProcess", process))
    assert channel.recv(timeout=1) == {"type": "hello", "mode": "session"}

    with pytest.raises(RuntimeError, match="before candidate materialization") as caught:
        channel.recv(timeout=1)

    assert not isinstance(caught.value, PiCandidateChannelError)
    channel.close()


def test_pre_session_protocol_failure_is_not_retyped_after_session_send() -> None:
    """A queued trusted-runner failure keeps its original ownership across phase changes."""
    process = _FakeProcess([])
    process.stdout = io.StringIO("A" * (mod._MAX_INBOUND_ENCODED_FRAME_CHARS + 1) + "\n")
    channel = LocalStdioChannel(cast("mod._TextProcess", process))

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with channel._inbound_lock:
            queued = list(channel._frames.queue)
            if any(isinstance(item, mod._ChannelFailure) for item, _size in queued):
                break
        time.sleep(0.001)
    else:
        pytest.fail("pre-session protocol failure was not queued")

    channel.send({"type": "session_start"})
    with pytest.raises(RuntimeError, match="before candidate materialization") as caught:
        channel.recv(timeout=1)

    assert not isinstance(caught.value, PiCandidateChannelError)
    channel.close()

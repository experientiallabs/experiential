"""Run the vendored pi live-session peer as a local Node.js process.

The platform remains the credential boundary: the Node peer receives worker
completions over the existing stdio frame protocol and never receives provider
keys. Unlike the E2B backend, this module deliberately runs the harness process
on the user's machine. The CLI presents an explicit consent prompt before it
reaches this boundary.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO, cast

from wmh.core.types import JsonObject
from wmh.harness.pi_e2b import HELLO_TIMEOUT_S, PI_NPM_PACKAGES, session_entry_files

if TYPE_CHECKING:
    from collections.abc import Callable

_PI_VERSION = "0.80.3"
_MIN_NODE = (22, 19, 0)
_PACKAGE_JSON = '{"name":"wmh-pi-local","private":true,"type":"module"}\n'
_INSTALL_MARKER = ".wmh-pi-dependencies"
_STDERR_LINES = 50


class _CompletedCommand(Protocol):
    """The subprocess result slice runtime bootstrap consumes."""

    stdout: str


class _TextProcess(Protocol):
    """The text-mode Popen slice used by the local frame channel."""

    stdin: TextIO | None
    stdout: TextIO | None
    stderr: TextIO | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class _Eof:
    """Reader-thread sentinel for a closed runner stdout stream."""


_EOF = _Eof()


def parse_node_version(output: str) -> tuple[int, int, int]:
    """Parse ``node --version`` output into a semantic-version triple."""
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)\s*", output)
    if match is None:
        raise RuntimeError(f"could not parse Node.js version output: {output.strip()!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def default_local_pi_runtime_dir() -> Path:
    """Return the user cache directory for the pinned local pi runtime."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "wmh" / "pi" / _PI_VERSION


def ensure_local_pi_runtime(
    runtime_dir: Path,
    *,
    node: str,
    npm: str,
    run_command: Callable[..., _CompletedCommand] = subprocess.run,
) -> Path:
    """Refresh the live runner and install its pinned npm dependencies once."""
    version = run_command(
        [node, "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    parsed = parse_node_version(version)
    if parsed < _MIN_NODE:
        required = ".".join(str(part) for part in _MIN_NODE)
        found = ".".join(str(part) for part in parsed)
        raise RuntimeError(f"local pi requires Node.js {required} or newer (found {found})")

    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "package.json").write_text(_PACKAGE_JSON, encoding="utf-8")
    for name, content in session_entry_files().items():
        (runtime_dir / name).write_text(content, encoding="utf-8")

    marker = runtime_dir / _INSTALL_MARKER
    expected = "\n".join(PI_NPM_PACKAGES) + "\n"
    if marker.is_file() and marker.read_text(encoding="utf-8") == expected:
        return runtime_dir
    run_command(
        [npm, "install", "--no-audit", "--no-fund", "--ignore-scripts", *PI_NPM_PACKAGES],
        cwd=runtime_dir,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    marker.write_text(expected, encoding="utf-8")
    return runtime_dir


class LocalStdioChannel:
    """A live-session frame channel over a local Node child process."""

    def __init__(
        self,
        process: _TextProcess,
        *,
        stderr_lines: int = _STDERR_LINES,
        cleanup_dir: Path | None = None,
    ) -> None:
        """Start bounded stdout/stderr reader threads for ``process``."""
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("local pi process must expose stdin, stdout, and stderr pipes")
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._stderr_stream = process.stderr
        self._frames: queue.Queue[JsonObject | _Eof] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._cleanup_dir = cleanup_dir
        self._closed = False
        threading.Thread(target=self._read_stdout, name="pi-local-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="pi-local-stderr", daemon=True).start()

    def send(self, frame: JsonObject) -> None:
        """Write one base64(JSON) frame to the child process."""
        line = base64.b64encode(json.dumps(frame).encode()).decode() + "\n"
        self._stdin.write(line)
        self._stdin.flush()

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        """Return the next decoded frame, optionally bounded by ``timeout``."""
        try:
            item = self._frames.get(timeout=timeout)
        except queue.Empty:
            message = f"no frame from local pi within {timeout}s{self._stderr_suffix()}"
            raise TimeoutError(message) from None
        if isinstance(item, _Eof):
            self._frames.put(item)
            if self._closed:
                return None
            raise RuntimeError(f"local pi process exited unexpectedly{self._stderr_suffix()}")
        return item

    def close(self) -> None:
        """Shut down the child process without leaving a local runner behind."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self.send({"type": "shutdown"})
        with contextlib.suppress(Exception):
            self._process.wait(timeout=2)
        if self._process.poll() is None:
            with contextlib.suppress(Exception):
                self._process.terminate()
            with contextlib.suppress(Exception):
                self._process.wait(timeout=2)
        if self._process.poll() is None:
            with contextlib.suppress(Exception):
                self._process.kill()
        if self._cleanup_dir is not None:
            with contextlib.suppress(OSError):
                shutil.rmtree(self._cleanup_dir)

    def _read_stdout(self) -> None:
        try:
            for raw in self._stdout:
                text = raw.strip()
                if not text:
                    continue
                try:
                    frame = json.loads(base64.b64decode(text, validate=True))
                except ValueError:
                    self._stderr.append(f"[stdout] {text}")
                    continue
                if isinstance(frame, dict):
                    self._frames.put(cast("JsonObject", frame))
                else:
                    self._stderr.append(f"[stdout] {text}")
        finally:
            self._frames.put(_EOF)

    def _read_stderr(self) -> None:
        for raw in self._stderr_stream:
            text = raw.rstrip()
            if text:
                self._stderr.append(text)

    def _stderr_suffix(self) -> str:
        tail = "\n".join(self._stderr)
        return f"; recent stderr:\n{tail}" if tail else ""


def start_local_live_runner(
    *,
    runtime_dir: Path | None = None,
    hello_timeout: float = HELLO_TIMEOUT_S,
) -> LocalStdioChannel:
    """Bootstrap and start the local pi peer, returning a hello-verified channel."""
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node is None or npm is None:
        raise RuntimeError("local pi requires Node.js 22.19+ and npm on PATH")
    root = ensure_local_pi_runtime(
        runtime_dir or default_local_pi_runtime_dir(), node=node, npm=npm
    )
    # Each process gets a private cwd for materialized champion code. Node still
    # resolves dependencies from the cached runner's parent directory.
    process_root = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed executable/args, no shell
            [node, "--experimental-strip-types", str(root / "runner_live.ts")],
            cwd=process_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except BaseException:
        shutil.rmtree(process_root, ignore_errors=True)
        raise
    channel = LocalStdioChannel(cast("_TextProcess", process), cleanup_dir=process_root)
    try:
        frame = channel.recv(timeout=hello_timeout)
        if frame is None or frame.get("type") != "hello":
            raise RuntimeError("local pi did not send its hello frame")
    except BaseException:
        channel.close()
        raise
    return channel

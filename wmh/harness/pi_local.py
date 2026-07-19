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
import gzip
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO, cast

from wmh.core.types import JsonObject
from wmh.harness.pi_e2b import (
    HELLO_TIMEOUT_S,
    PI_NPM_PACKAGES,
    TRANSPORT_KEEPALIVE_TYPE,
    session_entry_files,
)
from wmh.harness.pi_runner import PiCandidateChannelError, PiOutboundFrameTooLargeError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_PI_VERSION = "0.80.3"
_MIN_NODE = (22, 19, 0)
_PACKAGE_JSON = """{
  "name": "wmh-pi-local",
  "private": true,
  "type": "module",
  "dependencies": {
    "@earendil-works/pi-ai": "0.80.3",
    "ignore": "7.0.5",
    "typebox": "1.1.38",
    "yaml": "2.9.0"
  }
}
"""
_PACKAGE_LOCK_GZIP_B64 = (
    "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"
)
_PACKAGE_LOCK_SHA256 = "275e9108b1b1703040f1a8230def66e7f4335e286ea14d1f5bb0ff1502274051"
_INSTALL_MARKER = ".wmh-pi-dependencies"
_STDERR_LINES = 50
_MAX_OUTBOUND_FRAME_BYTES = 6 * 1024 * 1024
_MAX_INBOUND_FRAME_BYTES = 1 * 1024 * 1024
_MAX_INBOUND_ENCODED_FRAME_CHARS = 4 * ((_MAX_INBOUND_FRAME_BYTES + 2) // 3)
_MAX_STDERR_LINE_CHARS = 64 * 1024
_MAX_PENDING_FRAMES = 16
_MAX_PENDING_FRAME_BYTES = 2 * 1024 * 1024
PI_CONTAINER_INDEX_DIGEST = (
    "sha256:4a4884e8a44826194dff92ba316264f392056cbe243dcc9fd3551e71cea02b90"
)
PI_CONTAINER_PLATFORM = "linux/amd64"
PI_CONTAINER_IMAGE = (
    "node:22.19.0-bookworm-slim"
    "@sha256:cff78eb5aa1cf27dc2b6aeea9d31366415a43e9a9ea0ddec00d780b2b66fad0f"
)
_DIGEST_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-fA-F]{64}\Z")
_CONTAINER_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_CONTAINER_LABEL_VALUE = re.compile(r"[A-Za-z0-9_.:-]{1,512}\Z")
_CONTAINER_PLATFORM = re.compile(r"[a-z0-9_.-]{1,64}/[a-z0-9_.-]{1,64}\Z")
_CONTAINER_RUNTIME_DIR = "/opt/wmh-pi"
_CONTAINER_WORK_DIR = "/work"
_CONTAINER_RUNNER_COMMAND = f"""\
if ! ln -s {_CONTAINER_RUNTIME_DIR}/package.json {_CONTAINER_WORK_DIR}/package.json; then
  exit 125
fi
if ! ln -s {_CONTAINER_RUNTIME_DIR}/node_modules {_CONTAINER_WORK_DIR}/node_modules; then
  exit 125
fi
node --experimental-strip-types {_CONTAINER_RUNTIME_DIR}/runner_live.ts
status=$?
if [ "$status" -ge 125 ] && [ "$status" -le 127 ]; then
  exit 1
fi
exit "$status"
"""


class _CompletedCommand(Protocol):
    """The subprocess result slice runtime bootstrap consumes."""

    stdout: str


class _CommandResult(_CompletedCommand, Protocol):
    """Completed Docker CLI command used to prove container cleanup."""

    stderr: str
    returncode: int


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

    def __init__(self, *, candidate_controlled: bool) -> None:
        self.candidate_controlled = candidate_controlled


class _ChannelFailure:
    """Reader-thread sentinel for candidate output rejected at the host boundary."""

    def __init__(self, message: str, *, candidate_controlled: bool) -> None:
        self.message = message
        self.candidate_controlled = candidate_controlled


def parse_node_version(output: str) -> tuple[int, int, int]:
    """Parse ``node --version`` output into a semantic-version triple."""
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)\s*", output)
    if match is None:
        raise RuntimeError(f"could not parse Node.js version output: {output.strip()!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _container_package_lock() -> str:
    """Return and integrity-check the deterministic npm lock embedded with the runner."""
    compressed = base64.b64decode(_PACKAGE_LOCK_GZIP_B64, validate=True)
    content = gzip.decompress(compressed)
    actual = hashlib.sha256(content).hexdigest()
    if actual != _PACKAGE_LOCK_SHA256:
        raise RuntimeError(
            "embedded pi package lock digest mismatch: "
            f"expected {_PACKAGE_LOCK_SHA256}, got {actual}"
        )
    return content.decode("utf-8")


def validate_pi_container_image(image: str) -> None:
    """Reject mutable tags at the candidate-container trust boundary."""
    if _DIGEST_IMAGE.fullmatch(image) is None:
        raise ValueError("pi container image must be a digest-qualified OCI reference")


@contextlib.contextmanager
def _exclusive_runtime_lock(path: Path) -> Iterator[None]:
    """Serialize runtime publication with a crash-released operating-system file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
        self._frames: queue.Queue[tuple[JsonObject | _Eof | _ChannelFailure, int]] = queue.Queue(
            maxsize=_MAX_PENDING_FRAMES
        )
        self._inbound_lock = threading.Lock()
        self._pending_frame_bytes = 0
        self._stderr: deque[str] = deque(maxlen=stderr_lines)
        self._cleanup_dir = cleanup_dir
        self._closed = False
        self._candidate_control_active = False
        threading.Thread(target=self._read_stdout, name="pi-local-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="pi-local-stderr", daemon=True).start()

    def send(self, frame: JsonObject) -> None:
        """Write one base64(JSON) frame to the child process."""
        payload = json.dumps(frame).encode()
        if len(payload) > _MAX_OUTBOUND_FRAME_BYTES:
            raise PiOutboundFrameTooLargeError(
                "local pi outbound frame exceeds the "
                f"{_MAX_OUTBOUND_FRAME_BYTES}-byte transport limit"
            )
        line = base64.b64encode(payload).decode() + "\n"
        if frame.get("type") == "session_start":
            # Linearize ownership with reader-thread terminal snapshots: once this lock is
            # released after a successful flush, any newly observed output is candidate-caused.
            # A failed write never crosses that boundary and must remain infrastructure.
            with self._inbound_lock:
                try:
                    self._stdin.write(line)
                    self._stdin.flush()
                except OSError as exc:
                    raise self._unexpected_exit_error(candidate_controlled=False) from exc
                self._candidate_control_active = True
            return
        try:
            self._stdin.write(line)
            self._stdin.flush()
        except OSError as exc:
            raise self._unexpected_exit_error() from exc

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        """Return the next decoded frame, optionally bounded by ``timeout``."""
        try:
            item, frame_bytes = self._frames.get(timeout=timeout)
        except queue.Empty:
            message = f"no frame from local pi within {timeout}s{self._stderr_suffix()}"
            raise TimeoutError(message) from None
        if frame_bytes:
            with self._inbound_lock:
                self._pending_frame_bytes -= frame_bytes
        if isinstance(item, _Eof):
            with self._inbound_lock:
                self._frames.put_nowait((item, 0))
            if self._closed:
                return None
            raise self._unexpected_exit_error(candidate_controlled=item.candidate_controlled)
        if isinstance(item, _ChannelFailure):
            if item.candidate_controlled:
                raise PiCandidateChannelError(item.message)
            raise RuntimeError(
                "pi runner protocol failed before candidate materialization: " + item.message
            )
        return item

    def _unexpected_exit_error(
        self,
        *,
        candidate_controlled: bool | None = None,
    ) -> RuntimeError:
        """Type an unexpected peer exit as candidate-controlled once a session is sent."""
        code = self._process.poll()
        detail = f" with status {code}" if code is not None else ""
        if candidate_controlled is None:
            with self._inbound_lock:
                candidate_controlled = self._candidate_control_active
        if not candidate_controlled:
            return RuntimeError(
                f"pi runner exited before candidate materialization{detail}{self._stderr_suffix()}"
            )
        return PiCandidateChannelError(
            f"candidate runner exited unexpectedly{detail}{self._stderr_suffix()}"
        )

    def close(self) -> None:
        """Shut down the child process and fail if its termination cannot be proved."""
        if self._closed:
            return
        self._closed = True
        failures: list[str] = []
        with contextlib.suppress(Exception):
            self.send({"type": "shutdown"})
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        except Exception as exc:  # noqa: BLE001 - teardown continues before reporting all failures
            failures.append(f"graceful wait failed: {exc}")
        if self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception as exc:  # noqa: BLE001 - kill remains as the final cleanup attempt
                failures.append(f"terminate failed: {exc}")
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            except Exception as exc:  # noqa: BLE001 - kill remains as the final cleanup attempt
                failures.append(f"post-terminate wait failed: {exc}")
        if self._process.poll() is None:
            try:
                self._process.kill()
            except Exception as exc:  # noqa: BLE001 - report an unproven cleanup below
                failures.append(f"kill failed: {exc}")
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            except Exception as exc:  # noqa: BLE001 - report an unproven cleanup below
                failures.append(f"post-kill wait failed: {exc}")
        if self._cleanup_dir is not None:
            try:
                shutil.rmtree(self._cleanup_dir)
            except OSError as exc:
                failures.append(f"private runner directory cleanup failed: {exc}")
        if self._process.poll() is None:
            failures.append("runner process is still alive after kill")
        if failures:
            raise RuntimeError("local pi cleanup was not proved: " + "; ".join(failures))

    def _read_stdout(self) -> None:
        try:
            while raw := self._stdout.readline(_MAX_INBOUND_ENCODED_FRAME_CHARS + 2):
                if len(raw) > _MAX_INBOUND_ENCODED_FRAME_CHARS + 1 or (
                    not raw.endswith("\n") and len(raw) > _MAX_INBOUND_ENCODED_FRAME_CHARS
                ):
                    self._put_inbound(
                        self._channel_failure(
                            "local pi encoded frame exceeded the host transport limit"
                        )
                    )
                    return
                text = raw.strip()
                if not text:
                    continue
                try:
                    payload = base64.b64decode(text, validate=True)
                except ValueError:
                    self._stderr.append(f"[invalid stdout frame] {text[:1024]}")
                    continue
                if len(payload) > _MAX_INBOUND_FRAME_BYTES:
                    self._put_inbound(
                        self._channel_failure(
                            "local pi decoded frame exceeded the host transport limit"
                        )
                    )
                    return
                try:
                    frame = json.loads(payload)
                except ValueError:
                    self._stderr.append(f"[invalid stdout frame] {text[:1024]}")
                    continue
                if isinstance(frame, dict):
                    if frame.get("type") == TRANSPORT_KEEPALIVE_TYPE:
                        continue
                    if not self._put_inbound(
                        cast("JsonObject", frame),
                        frame_bytes=len(payload),
                    ):
                        return
                else:
                    self._stderr.append(f"[invalid stdout frame] {text[:1024]}")
        finally:
            self._put_inbound(_Eof(candidate_controlled=self._candidate_phase_snapshot()))

    def _read_stderr(self) -> None:
        while raw := self._stderr_stream.readline(_MAX_STDERR_LINE_CHARS + 1):
            if len(raw) > _MAX_STDERR_LINE_CHARS:
                self._stderr.append("[stderr line truncated at host transport limit]")
                continue
            text = raw.rstrip()
            if text:
                self._stderr.append(text)

    def _put_inbound(
        self,
        item: JsonObject | _Eof | _ChannelFailure,
        *,
        frame_bytes: int = 0,
    ) -> bool:
        """Enqueue one item or replace a candidate flood with a bounded failure sentinel."""
        if self._closed:
            return False
        with self._inbound_lock:
            if self._closed:
                return False
            if self._pending_frame_bytes + frame_bytes > _MAX_PENDING_FRAME_BYTES:
                self._replace_pending_with_failure(
                    _ChannelFailure(
                        "candidate runner exceeded the host pending-frame byte budget",
                        candidate_controlled=self._candidate_control_active,
                    )
                )
                return False
            if self._frames.full():
                failure = (
                    item
                    if isinstance(item, _ChannelFailure)
                    else _ChannelFailure(
                        "candidate runner exceeded the host pending-frame count budget",
                        candidate_controlled=self._candidate_control_active,
                    )
                )
                self._replace_pending_with_failure(failure)
                return False
            self._frames.put_nowait((item, frame_bytes))
            self._pending_frame_bytes += frame_bytes
            return True

    def _replace_pending_with_failure(self, failure: _ChannelFailure) -> None:
        """Drop queued candidate data and publish one zero-byte terminal failure under lock."""
        while True:
            try:
                _item, frame_bytes = self._frames.get_nowait()
            except queue.Empty:
                break
            self._pending_frame_bytes -= frame_bytes
        self._frames.put_nowait((failure, 0))

    def _candidate_phase_snapshot(self) -> bool:
        """Capture which side controlled the runner when a terminal condition was observed."""
        with self._inbound_lock:
            return self._candidate_control_active

    def _channel_failure(self, message: str) -> _ChannelFailure:
        """Create a protocol sentinel without allowing later phase changes to retype it."""
        return _ChannelFailure(
            message,
            candidate_controlled=self._candidate_phase_snapshot(),
        )

    def _stderr_suffix(self) -> str:
        tail = "\n".join(self._stderr)
        return f"; recent stderr:\n{tail}" if tail else ""


class DockerStdioChannel(LocalStdioChannel):
    """Local frame channel that proves its Docker container no longer exists on close."""

    def __init__(
        self,
        process: _TextProcess,
        *,
        docker: str,
        container_name: str,
        cleanup_dir: Path | None,
        control_dir: Path,
        run_command: Callable[..., _CommandResult] = subprocess.run,
    ) -> None:
        super().__init__(process, cleanup_dir=cleanup_dir)
        self._docker = docker
        self._container_name = container_name
        self._control_dir = control_dir
        self._run_command = run_command
        self._container_cleanup_proved = False
        self._close_lock = threading.Lock()
        self._fully_closed = False

    @property
    def container_id(self) -> str:
        """Return the daemon resource identity, falling back to its unique name."""
        cid_file = self._control_dir / "container.cid"
        container_id = cid_file.read_text(encoding="utf-8").strip() if cid_file.is_file() else ""
        if re.fullmatch(r"[0-9a-fA-F]{12,64}", container_id) is not None:
            return container_id.lower()
        return self._container_name

    def close(self) -> None:
        """Close the client stream, force-remove the container, and verify daemon state."""
        with self._close_lock:
            if self._fully_closed:
                return
            self._close_and_prove()

    def _unexpected_exit_error(
        self,
        *,
        candidate_controlled: bool | None = None,
    ) -> RuntimeError:
        """Keep Docker client/invocation failures distinct from candidate exits."""
        code = self._process.poll()
        if code is not None and (code < 0 or code in {125, 126, 127}):
            return RuntimeError(
                f"Docker pi runner failed with status {code}{self._stderr_suffix()}"
            )
        return super()._unexpected_exit_error(candidate_controlled=candidate_controlled)

    def _close_and_prove(self) -> None:
        """Perform one serialized cleanup attempt."""
        failures: list[str] = []
        try:
            super().close()
        except Exception as exc:  # noqa: BLE001 - daemon cleanup must still run
            failures.append(str(exc))
        if not self._container_cleanup_proved:
            try:
                cid_file = self._control_dir / "container.cid"
                container_id = (
                    cid_file.read_text(encoding="utf-8").strip() if cid_file.is_file() else ""
                )
                target = (
                    container_id
                    if re.fullmatch(r"[0-9a-fA-F]{12,64}", container_id) is not None
                    else self._container_name
                )
                self._run_command(
                    [self._docker, "container", "rm", "--force", target],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                inspected = self._run_command(
                    [self._docker, "container", "inspect", target],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                detail = (inspected.stderr or inspected.stdout).strip()
                missing = (
                    "no such object" in detail.lower() or "no such container" in detail.lower()
                )
                if inspected.returncode == 0:
                    failures.append(
                        f"candidate container {self._container_name!r} still exists after removal"
                    )
                elif not missing:
                    failures.append(
                        "could not prove candidate container removal through Docker: "
                        f"{detail or f'inspect exited {inspected.returncode}'}"
                    )
                else:
                    self._container_cleanup_proved = True
            except Exception as exc:  # noqa: BLE001 - absence must be proved, never assumed
                failures.append(f"could not prove candidate container removal: {exc}")
        if self._process.poll() is None:
            failures.append("Docker client process is still alive after container removal")
        if self._control_dir.exists():
            try:
                shutil.rmtree(self._control_dir)
            except OSError as exc:
                failures.append(f"container control directory cleanup failed: {exc}")
        if failures:
            raise RuntimeError("isolated pi cleanup failed: " + "; ".join(failures))
        self._fully_closed = True


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


def default_container_pi_runtime_dir() -> Path:
    """Return the cache directory for Linux pi dependencies used by the container runner."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "wmh" / "pi-container" / _PI_VERSION


def ensure_container_pi_runtime(
    runtime_dir: Path,
    *,
    docker: str,
    image: str = PI_CONTAINER_IMAGE,
    platform: str = PI_CONTAINER_PLATFORM,
    run_command: Callable[..., _CompletedCommand] = subprocess.run,
) -> Path:
    """Atomically publish one immutable, content-addressed pi runtime namespace."""
    validate_pi_container_image(image)
    validate_pi_container_platform(platform)
    cache_root = runtime_dir.expanduser().resolve()
    package_lock = _container_package_lock()
    entry_files = session_entry_files()
    source_fingerprint = container_pi_bundle_digest().removeprefix("sha256:")
    image_identity = "sha256:" + image.rsplit("@sha256:", 1)[1].lower()
    namespace_input = json.dumps(
        {
            "image": image_identity,
            "platform": platform,
            "source": f"sha256:{source_fingerprint}",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    namespace = hashlib.sha256(namespace_input.encode()).hexdigest()
    published_runtime = cache_root / namespace
    lock_path = cache_root / f".{namespace}.lock"
    expected = f"{image_identity}\nsha256:{source_fingerprint}\n"
    published_files = {
        "package.json": _PACKAGE_JSON,
        "package-lock.json": package_lock,
        **entry_files,
    }

    with _exclusive_runtime_lock(lock_path):
        if published_runtime.is_symlink() or (
            published_runtime.exists() and not published_runtime.is_dir()
        ):
            raise RuntimeError(f"immutable pi runtime namespace is unsafe: {published_runtime}")
        marker = published_runtime / _INSTALL_MARKER
        try:
            current = marker.is_file() and marker.read_text(encoding="utf-8") == expected
            current = current and (published_runtime / "node_modules").is_dir()
            current = current and all(
                (published_runtime / name).is_file()
                and (published_runtime / name).read_text(encoding="utf-8") == content
                for name, content in published_files.items()
            )
        except OSError:
            current = False
        if current:
            return published_runtime
        if published_runtime.exists():
            raise RuntimeError(
                f"immutable pi runtime namespace is incomplete or corrupted: {published_runtime}"
            )

        staging_prefix = f".{namespace}.staging-"
        for orphan in cache_root.glob(f"{staging_prefix}*"):
            if orphan.is_symlink() or not orphan.is_dir():
                raise RuntimeError(f"immutable pi runtime staging path is unsafe: {orphan}")
            shutil.rmtree(orphan)

        staging_dir = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=cache_root))
        try:
            for name, content in published_files.items():
                destination = staging_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
            run_command(
                [
                    docker,
                    "run",
                    "--rm",
                    "--platform",
                    platform,
                    "--log-driver",
                    "none",
                    "--network",
                    "bridge",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--mount",
                    f"type=bind,src={staging_dir},dst={_CONTAINER_RUNTIME_DIR}",
                    "--workdir",
                    _CONTAINER_RUNTIME_DIR,
                    image,
                    "npm",
                    "ci",
                    "--no-audit",
                    "--no-fund",
                    "--ignore-scripts",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if not (staging_dir / "node_modules").is_dir():
                raise RuntimeError("pi container dependency install did not publish node_modules")
            marker = staging_dir / _INSTALL_MARKER
            with marker.open("w", encoding="utf-8") as handle:
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
            staging_dir.rename(published_runtime)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
    return published_runtime


def start_container_live_runner(
    *,
    runtime_dir: Path | None = None,
    image: str = PI_CONTAINER_IMAGE,
    platform: str = PI_CONTAINER_PLATFORM,
    labels: dict[str, str] | None = None,
    hello_timeout: float = HELLO_TIMEOUT_S,
    run_command: Callable[..., _CommandResult] = subprocess.run,
) -> LocalStdioChannel:
    """Start pi in an isolated local container with no host credentials or network."""
    validate_pi_container_image(image)
    validate_pi_container_platform(platform)
    validated_labels = _validate_container_labels(labels or {})
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("the local isolated pi runner requires Docker on PATH")
    root = ensure_container_pi_runtime(
        runtime_dir or default_container_pi_runtime_dir(),
        docker=docker,
        image=image,
        platform=platform,
        run_command=run_command,
    )
    control_root = Path(tempfile.mkdtemp(prefix="wmh-pi-container-control-"))
    container_name = f"wmh-pi-{uuid.uuid4().hex}"
    command = [
        docker,
        "run",
        "--rm",
        "--interactive",
        "--init",
        "--platform",
        platform,
        "--log-driver",
        "none",
        "--name",
        container_name,
        "--cidfile",
        str(control_root / "container.cid"),
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "1g",
        "--memory-swap",
        "1g",
        "--cpus",
        "2",
        "--ulimit",
        "nofile=1024:1024",
        "--ulimit",
        "core=0:0",
        "--user",
        "65534:65534",
        "--env",
        "HOME=/tmp",
        "--mount",
        f"type=bind,src={root},dst={_CONTAINER_RUNTIME_DIR},readonly",
        "--tmpfs",
        f"{_CONTAINER_WORK_DIR}:rw,nosuid,nodev,noexec,size=256m,mode=0700,uid=65534,gid=65534",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m",
        "--workdir",
        _CONTAINER_WORK_DIR,
    ]
    for key, value in sorted(validated_labels.items()):
        command.extend(("--label", f"{key}={value}"))
    command.extend((image, "/bin/sh", "-c", _CONTAINER_RUNNER_COMMAND))
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed Docker argv, no shell
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except BaseException:
        shutil.rmtree(control_root, ignore_errors=True)
        raise
    channel = DockerStdioChannel(
        cast("_TextProcess", process),
        docker=docker,
        container_name=container_name,
        cleanup_dir=None,
        control_dir=control_root,
        run_command=run_command,
    )
    try:
        frame = channel.recv(timeout=hello_timeout)
        if frame is None or frame.get("type") != "hello":
            raise RuntimeError("container pi did not send its hello frame")
    except BaseException:
        channel.close()
        raise
    return channel


def verify_container_pi_runner_ready(
    *,
    image: str = PI_CONTAINER_IMAGE,
    platform: str = PI_CONTAINER_PLATFORM,
    hello_timeout: float = HELLO_TIMEOUT_S,
) -> None:
    """Prove the isolated local runner can bootstrap, start, and clean up."""
    with contextlib.closing(
        start_container_live_runner(
            image=image,
            platform=platform,
            hello_timeout=hello_timeout,
        )
    ):
        pass


def container_pi_bundle_digest() -> str:
    """Return the canonical digest of every file mounted into the Pi runner."""
    source_manifest = json.dumps(
        {
            "entry_files": dict(sorted(session_entry_files().items())),
            "package_json": _PACKAGE_JSON,
            "package_lock": _container_package_lock(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return "sha256:" + hashlib.sha256(source_manifest).hexdigest()


def validate_pi_container_platform(platform: str) -> None:
    """Reject implicit or malformed local runner platforms."""
    if _CONTAINER_PLATFORM.fullmatch(platform) is None:
        raise ValueError("pi container platform must use lowercase os/architecture form")


def _validate_container_labels(labels: dict[str, str]) -> dict[str, str]:
    """Validate bounded Docker labels before adding them to argv."""
    for key, value in labels.items():
        if _CONTAINER_LABEL.fullmatch(key) is None:
            raise ValueError("pi container label key is invalid")
        if _CONTAINER_LABEL_VALUE.fullmatch(value) is None:
            raise ValueError("pi container label value is invalid")
    return dict(labels)


def reap_container_runner_lease(
    lease_id: str,
    *,
    run_command: Callable[..., _CommandResult] = subprocess.run,
) -> tuple[str, ...]:
    """Remove and then prove absence of every container carrying one lease label."""
    if _CONTAINER_LABEL_VALUE.fullmatch(lease_id) is None:
        raise ValueError("pi container lease identity is invalid")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("the local isolated pi runner requires Docker on PATH")
    label = f"wmh.runner.lease={lease_id}"
    listed = run_command(
        [docker, "container", "ls", "--all", "--quiet", "--filter", f"label={label}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    resource_ids = tuple(line.strip() for line in listed.stdout.splitlines() if line.strip())
    if any(re.fullmatch(r"[0-9a-fA-F]{12,64}", item) is None for item in resource_ids):
        raise RuntimeError("Docker returned an invalid orphan container identity")
    for resource_id in resource_ids:
        run_command(
            [docker, "container", "rm", "--force", resource_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    remaining = run_command(
        [docker, "container", "ls", "--all", "--quiet", "--filter", f"label={label}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if remaining.stdout.strip():
        raise RuntimeError("Docker runner orphan cleanup could not prove resource absence")
    return resource_ids

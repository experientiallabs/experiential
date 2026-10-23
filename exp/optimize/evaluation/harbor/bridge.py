"""Trial-scoped HTTP transport without actor access to grading evidence or source stores."""

import json
import os
import secrets
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import ThreadingUnixStreamServer
from threading import Lock, Thread, current_thread
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall


class HarborTrialOperations(Protocol):
    """Execute-only actor and private grading operations shared by frozen runtime owners."""

    def tools(self) -> JsonObject:
        """Return the public tool catalog."""
        ...

    def call(self, action: ToolCall) -> JsonObject:
        """Execute or replay one actor operation."""
        ...

    def submit(self, answer: JsonObject) -> JsonObject:
        """Freeze the final actor submission."""
        ...

    def result(self) -> JsonObject:
        """Return the privately evaluated outcome."""
        ...

    def output_files(self) -> JsonObject:
        """Return only explicitly declared public output files."""
        ...


@dataclass(frozen=True)
class HarborEndpoint:
    """Runtime-only capabilities, excluded from task packages and receipts.

    Attributes:
        url: Endpoint for exactly one trial, configured into the selected Harbor backend.
        actor_token: Execute/submit permission, never access to private grading.
        verifier_token: Private grading permission, never exposed to the actor.
    """

    url: str
    actor_token: str = field(repr=False)
    verifier_token: str = field(repr=False)
    socket_path: Path | None = None


class _UnixServer(ThreadingUnixStreamServer):
    """Own accepted sockets and threads with truthful, bounded shutdown."""

    def __init__(self, path: str, handler: type[BaseHTTPRequestHandler]) -> None:
        self.workers: dict[Thread, socket.socket] = {}
        self.worker_lock = Lock()
        super().__init__(path, handler, bind_and_activate=False)

    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: object
    ) -> None:
        """Register before starting so close cannot miss an accepted worker."""
        if not isinstance(request, socket.socket):
            raise TypeError("Unix Harbor transport requires a stream socket")
        worker = Thread(target=self.process_request_thread, args=(request, client_address))
        with self.worker_lock:
            self.workers[worker] = request
        try:
            worker.start()
        except BaseException:
            with self.worker_lock:
                self.workers.pop(worker)
            self.shutdown_request(request)
            raise

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: object
    ) -> None:
        """Keep the existing request handler and always retire its ownership record."""
        try:
            super().process_request_thread(request, str(client_address))
        finally:
            with self.worker_lock:
                self.workers.pop(current_thread(), None)

    def server_close(self) -> None:
        """Interrupt blocked reads and refuse a false cleanup claim for stuck backend work."""
        # Avoid ThreadingMixIn's unbounded join; these exact workers are owned here instead.
        self.block_on_close = False
        super().server_close()
        with self.worker_lock:
            workers = tuple(self.workers.items())
        for _, request in workers:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = time.monotonic() + 5
        for worker, _ in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        if any(worker.is_alive() for worker, _ in workers):
            raise TimeoutError("Unix Harbor backend workers did not close within five seconds")


@contextmanager
def serve_harbor_trial(
    trial: HarborTrialOperations,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    socket_path: Path | None = None,
) -> Iterator[HarborEndpoint]:
    """Expose one existing session and close transport on exit.

    Args:
        trial: Trusted application-owned session and evaluator.
        host: Explicit bind address. Defaults to loopback, never a public deployment.
        port: Explicit port, or zero to allocate a free local port.

    Yields:
        Runtime capabilities to inject separately into actor and verifier environments.
    """
    if socket_path is not None:
        if host != "127.0.0.1" or port != 0:
            raise ValueError("Unix transport does not accept a TCP host or port")
        parent = socket_path.parent
        if (
            not socket_path.is_absolute()
            or parent.resolve() != parent
            or not parent.is_dir()
            or parent.stat().st_uid != os.geteuid()
            or parent.stat().st_mode & 0o022
            or socket_path.exists()
            or socket_path.is_symlink()
            or any(parent.iterdir())
        ):
            raise ValueError(
                "Unix backend requires a fresh owned socket-only directory and absent path"
            )
    actor_token, verifier_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        """Transport boundary with no endpoint for arbitrary files or simulator internals."""

        def setup(self) -> None:
            """Bound incomplete HTTP bodies so closing a trial can drain request handlers."""
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format: str, *args: str) -> None:
            """Suppress request metadata that may contain runtime capabilities."""

        def do_GET(self) -> None:
            """Return public tool descriptions only for an authorized actor."""
            if self.path != "/tools":
                self.reply(404, {"error": "unknown endpoint"})
            elif self.authorized(actor_token):
                self.reply(200, trial.tools())

        def do_POST(self) -> None:
            """Dispatch fixed typed operations, without actor-selected paths or code."""
            if self.path not in ("/call", "/submit", "/grade", "/outputs"):
                self.reply(404, {"error": "unknown endpoint"})
                return
            if not self.authorized(verifier_token if self.path == "/grade" else actor_token):
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 131072:
                    raise ValueError("request exceeds transport allowance")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
                if self.path == "/call":
                    result = trial.call(ToolCall.model_validate(value))
                elif self.path == "/submit":
                    result = trial.submit(value)
                elif self.path == "/outputs":
                    if value:
                        raise ValueError("output retrieval takes no actor-provided evidence")
                    result = trial.output_files()
                elif value:
                    raise ValueError("grading takes no actor-provided evidence")
                else:
                    result = trial.result()
                self.reply(200, result)
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": type(error).__name__})
            except Exception as error:  # noqa: BLE001 - retain failure in authoritative owners.
                self.reply(503, {"error": type(error).__name__, "status": "incomplete"})

        def authorized(self, token: str) -> bool:
            """Separate actor and verifier capability scopes before reading the body."""
            if secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                return True
            self.reply(403, {"error": "capability does not authorize this operation"})
            return False

        def reply(self, status: int, value: JsonObject) -> None:
            """Return bounded JSON without traces, filesystem paths or credentials."""
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = (
        ThreadingHTTPServer((host, port), Handler)
        if socket_path is None
        else _UnixServer(str(socket_path), Handler)
    )
    server.daemon_threads = False
    server.block_on_close = True
    owned_inode = None
    thread = Thread(target=server.serve_forever, daemon=True)
    started = False
    closed = False
    try:
        if socket_path is not None:
            server.server_bind()
            owned_inode = socket_path.lstat()
            socket_path.chmod(0o666)
            server.server_activate()
        thread.start()
        started = True
        url = (
            f"http://{host}:{server.server_port}"
            if isinstance(server, ThreadingHTTPServer)
            else "http://localhost"
        )
        yield HarborEndpoint(url, actor_token, verifier_token, socket_path)
    finally:
        try:
            if started:
                server.shutdown()
            server.server_close()
            closed = True
        finally:
            if started:
                thread.join(timeout=2)
            if closed and socket_path is not None and owned_inode is not None:
                current = socket_path.lstat() if socket_path.exists() else None
                if current is not None and (current.st_dev, current.st_ino) == (
                    owned_inode.st_dev,
                    owned_inode.st_ino,
                ):
                    socket_path.unlink()

"""Trial-scoped HTTP transport without actor access to grading evidence or source stores."""

import json
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.optimize.evaluation.harbor_trial import HarborTrial


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


@contextmanager
def serve_harbor_trial(
    trial: HarborTrial, *, host: str = "127.0.0.1", port: int = 0
) -> Iterator[HarborEndpoint]:
    """Expose one existing session and close transport on exit.

    Args:
        trial: Trusted application-owned session and evaluator.
        host: Explicit bind address. Defaults to loopback, never a public deployment.
        port: Explicit port, or zero to allocate a free local port.

    Yields:
        Runtime capabilities to inject separately into actor and verifier environments.
    """
    actor_token, verifier_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        """Transport boundary with no endpoint for arbitrary files or simulator internals."""

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
            if self.path not in ("/call", "/submit", "/grade"):
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

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield HarborEndpoint(f"http://{host}:{server.server_port}", actor_token, verifier_token)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

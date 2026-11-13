"""Serve scripted test detectors over the same HTTP contract Rust uses in production."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.client import InspectingClassifier
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCompletion,
)

_endpoints: list[NativeClassifierFixture] = []


class NativeClassifierFixture:
    """Expose a fixture model as a loopback service, with no gateway callback fallback."""

    def __init__(self, classifier: InspectingClassifier) -> None:
        """Keep the existing scripted model and create HTTP resources only if requested."""
        self.classifier = classifier
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.checks: dict[str, GuardrailCheck] = {}

    def configuration(self) -> JsonObject:
        """Bind an owned loopback endpoint and export its explicit native egress authority."""
        with self.lock:
            if self.server is None:
                fixture = self

                class Handler(BaseHTTPRequestHandler):
                    """Run the synthetic decision model independently of gateway threads."""

                    def do_POST(self) -> None:  # noqa: N802 - HTTP protocol.
                        """Validate the standard classifier envelope and emit a typed verdict."""
                        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        check = fixture.checks.get(payload["check_id"]) or GuardrailCheck(
                            check_id=payload["check_id"],
                            capability=payload["capability"],
                            stage=payload["stage"],
                            action=payload["action"],
                            adapter_id="fixture",
                            timeout_ms=5000,
                        )
                        try:
                            if payload["stage"] == "input":
                                request = GatewayRequest.model_validate(payload["request"])
                                operation = fixture.classifier.inspect_input(
                                    request=request, check=check
                                )
                            else:
                                completion = GuardrailCompletion.model_validate(
                                    payload["completion"]
                                )
                                operation = fixture.classifier.inspect_output(
                                    completion=completion, check=check
                                )
                            verdict = asyncio.run(
                                _bounded(operation, check.timeout_ms / 1000 + 0.25)
                            )
                            body = verdict.model_dump_json().encode()
                            status = 200
                        except Exception:  # noqa: BLE001 - model failures are real HTTP uncertainty.
                            body, status = b'{"error":"fixture unavailable"}', 503
                        try:
                            self.send_response(status)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(body)))
                            self.end_headers()
                            self.wfile.write(body)
                        except (BrokenPipeError, ConnectionResetError):
                            pass

                    def log_message(self, format: str, *args: object) -> None:
                        """Keep synthetic classifier requests out of diagnostics."""

                self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                self.server.daemon_threads = True
                self.thread = threading.Thread(
                    target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
                )
                self.thread.start()
                _endpoints.append(self)
            return {
                "kind": "http_json",
                "url": f"http://127.0.0.1:{self.server.server_port}",
                "max_response_bytes": 1_048_576,
                "transport": {"public_only": False, "headers": {}},
            }

    def native_input_configuration(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> JsonObject:
        """Freeze a complete initial input for Rust's HTTP client."""
        self.checks[check.check_id] = check
        return self.configuration() | {
            "payload": {
                "request": request.model_dump(mode="json"),
                "check_id": check.check_id,
                "capability": check.capability.value,
                "action": check.action.value,
                "stage": "input",
            }
        }

    def native_input_template(self, *, check: GuardrailCheck) -> JsonObject:
        """Let native reinspection rebuild the same service's envelope after tool expansion."""
        self.checks[check.check_id] = check
        return self.configuration()

    def native_output_configuration(self) -> JsonObject:
        """Export a complete-output model endpoint without executing it on admission."""
        return self.configuration()

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Preserve the original fixture contract for Python-only engine unit tests."""
        return await self.classifier.inspect_input(request=request, check=check)

    async def inspect_output(
        self, *, completion: GuardrailCompletion, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Preserve the original fixture contract outside native serving."""
        return await self.classifier.inspect_output(completion=completion, check=check)

    def close(self) -> None:
        """Release this fixture's listener without waiting on deliberately stalled model calls."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(2)
            assert not self.thread.is_alive()


async def _bounded(operation: Awaitable[ClassifierVerdict], timeout: float) -> ClassifierVerdict:
    """Bound the synthetic model independently of Rust's own request cancellation."""
    return await asyncio.wait_for(operation, timeout)


def close_classifier_fixtures() -> None:
    """Drain only listeners created by the completed test in this worker process."""
    while _endpoints:
        _endpoints.pop().close()

"""Race each gateway tool successor against its own native input inspection."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import httpx
import pytest

from exp.common.sqlite.native import connect as connect_sqlite
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.tests.guardrail_observation_test import _policy
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_tool_search_test import (
    _TOOLS_CHAT,
    _configure,
    _search_call_turn,
    _text_turn,
)
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _NativeOnlyControl


class _Rounds:
    """Answer the first call immediately and independently hold the successor."""

    def __init__(self, *, provider: bool, blocked: bool = False, status: int = 200) -> None:
        """Start a synthetic server with explicit second-call synchronization."""
        self.started = threading.Event()
        self.release = threading.Event()
        self.requests: list[dict[str, object]] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            """Return either provider SSE or an input inspection verdict."""

            def do_POST(self) -> None:  # noqa: N802
                """Keep the successor separate from the initial approved input."""
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(body)
                subject = body if provider else body["request"]
                followup = any(message.get("role") == "tool" for message in subject["messages"])
                if followup:
                    endpoint.started.set()
                    if not endpoint.release.wait(8):
                        return
                payload = (
                    (_text_turn() if followup else _search_call_turn("tool_search"))
                    if provider
                    else json.dumps({"flagged": followup and blocked}).encode()
                )
                try:
                    self.send_response(status if followup else 200)
                    self.send_header(
                        "Content-Type", "text/event-stream" if provider else "application/json"
                    )
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: object) -> None:
                """Keep synthetic request contents out of logs."""

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        """Release bounded handlers and stop the server."""
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("outcome", ["allow", "block", "error", "grace"])
def test_tool_successor_has_an_independent_parallel_decision(
    tmp_path: Path, stream: bool, outcome: str
) -> None:
    """A prior approval never authorizes a denied successor or serializes its dispatch."""
    _round_case(tmp_path, stream=stream, outcome=outcome, execution="parallel")


@pytest.mark.parametrize("outcome", ["allow", "block", "error"])
def test_serial_tool_successor_waits_for_its_own_decision(tmp_path: Path, outcome: str) -> None:
    """Serial mode withholds third-party dispatch, while detector failure still opens it."""
    _round_case(tmp_path, stream=False, outcome=outcome, execution="before_dispatch")


def _round_case(
    tmp_path: Path, *, stream: bool, outcome: str, execution: Literal["parallel", "before_dispatch"]
) -> None:
    """Exercise real native accounting, sockets, and persisted per-inspection outcomes."""
    provider = _Rounds(provider=True)
    moderator = _Rounds(
        provider=False,
        blocked=outcome in {"block", "grace"},
        status=503 if outcome == "error" else 200,
    )
    key = _configure(tmp_path, provider.url + "/v1")
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
    policy = _policy(timeout_ms=4000).model_copy(
        update={
            "mode": "enforce",
            "input_execution": execution,
            "input_fail_open": True,
            "input_max_wait_ms": 40 if outcome == "grace" else 1000,
        }
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"detector": _Classifier(moderator)})),
        monotonic=time.monotonic,
    )
    control = _NativeOnlyControl(components, guardrails=engine)
    try:
        with _serving(control) as url, ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                httpx.post,
                url + "/v1/chat/completions",
                headers={"authorization": f"Bearer {key}"},
                json={
                    "model": "coding",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": _TOOLS_CHAT,
                    "stream": stream,
                },
                timeout=6,
            )
            try:
                assert moderator.started.wait(2)
                if execution == "parallel":
                    assert provider.started.wait(2), (
                        "follow-up generation serialized behind inspection"
                    )
                else:
                    assert not provider.started.is_set()
                started = time.monotonic()
                if outcome != "grace":
                    moderator.release.set()
                if outcome != "block":
                    provider.release.set()
                response = result.result(timeout=2)
                assert response.status_code == (400 if outcome == "block" else 200), response.text
                if outcome == "block":
                    assert not provider.release.is_set()
                    assert "18C in Bern" not in response.text
                else:
                    assert "18C in Bern" in response.text
                if outcome == "grace":
                    assert time.monotonic() - started < 0.75
                    assert not moderator.release.is_set()
            finally:
                provider.release.set()
                moderator.release.set()
        assert control.close(2)
        with connect_sqlite(components.ledger.database_path) as db:
            records = [
                json.loads(row[0])
                for row in db.execute("SELECT result FROM gateway_input_moderation")
            ]
        assert len(records) == (2 if execution == "parallel" else 3)
        assert any(
            record["status"]
            == (
                "denied"
                if outcome == "block"
                else "failed_open"
                if outcome in {"error", "grace"}
                else "allowed"
            )
            for record in records
        )
        assert len(provider.requests) == (
            1 if execution == "before_dispatch" and outcome == "block" else 2
        )
    finally:
        provider.release.set()
        moderator.release.set()
        control.close(2)
        engine.close(timeout_seconds=2)
        components.write_ledger.close()
        provider.close()
        moderator.close()

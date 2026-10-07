"""Exercise real native OpenAI Decisions API dispatch and SQLite settlement on loopback.

Only the OpenAI-shaped HTTP upstream is synthetic. The shared serving driver runs
the Axum ``/v1/decisions`` route, Rust admission/dispatch and the Python control
plane over a real SQLite authority, attempt ledger and monthly budget. A
driver-local wire profile replacement points only OpenAI's decisions endpoint at
the loopback server; the production client and its URL derivation stay intact.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
    NormalizedGatewayCatalog,
)
from exp.common.models.gateway_pools import GatewayEquivalenceCertification
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.certified_pool_fixture_test import upsert_certified_pool
from exp.runtime.gateway.tests.native_messages_test import _DRIVER_SOURCE, _HOST, _ServingEngine

pytest.importorskip("exp_gateway_native")

_TIMEOUT_SECONDS = 8.0
_PROVIDER_KEY = "openai-loopback-secret-canary"
_INPUT_RATE = 100_000_000

_PROFILE_OVERRIDE_SOURCE = textwrap.dedent(
    '''
    """Keep the production OpenAI client; point only its decisions endpoint at loopback."""

    import json
    import sys
    from dataclasses import replace

    from exp.runtime.models.providers.base import GatewayWireProfile
    from exp.runtime.models.providers.openai import OpenAIClient

    _original_openai_profile = OpenAIClient.gateway_wire_profile
    _loopback_url = json.loads(sys.argv[1])["openai_decisions_loopback_url"]


    def _loopback_profile(self: OpenAIClient) -> GatewayWireProfile:
        """Replace only the official decisions URL after real construction."""
        profile = _original_openai_profile(self)
        if profile.openai_decisions_url != "https://api.openai.com/v1/decisions":
            return profile
        return replace(profile, openai_decisions_url=_loopback_url)


    OpenAIClient.gateway_wire_profile = _loopback_profile
    '''
).strip()

_USAGE: JsonObject = {
    "input_tokens": 275,
    "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 275,
}


def _body(selector: str = "answer", model: str = "luna-decisions") -> JsonObject:
    """Build every question type; the input text selects the upstream behavior."""
    return {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": selector}],
        "questions": [
            {
                "type": "choice",
                "name": "intent",
                "instructions": "What does the customer want?",
                "choices": [
                    {"value": "refund", "description": "Money back"},
                    {"value": True},
                ],
            },
            {"type": "predicate", "instructions": "Is the customer angry?"},
            {
                "type": "score",
                "name": "urgency",
                "instructions": "How urgent?",
                "levels": [{"label": "low"}, {"label": "high", "description": "now"}],
            },
        ],
        "safety_identifier": "end-user-1",
    }


def _answers(selector: str) -> list[JsonObject]:
    """Return answers in question order, declining the score for ``refusal``."""
    answers: list[JsonObject] = [
        {
            "type": "choice",
            "name": "intent",
            "choice": "refund",
            "probabilities": [
                {"value": "refund", "probability": 0.9},
                {"value": True, "probability": 0.1},
            ],
            "confidence": 0.9,
        },
        {"type": "predicate", "probability": 0.31},
        {
            "type": "score",
            "name": "urgency",
            "score": 0.7,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.3},
                {"value": 1, "label": "high", "probability": 0.7},
            ],
            "confidence": 0.7,
        },
    ]
    if selector == "refusal":
        answers[2] = {"type": "refusal", "name": "urgency"}
    if selector == "malformed":
        answers[0]["choice"] = "true"
    return answers


class _DecisionsUpstream(BaseHTTPRequestHandler):
    """Serve bounded synthetic OpenAI answers, recording loopback traffic only."""

    payloads: list[JsonObject] = []
    headers_seen: list[dict[str, str]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer, reject the primary credential, or corrupt one answer."""
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        with self.lock:
            self.payloads.append(payload)
            self.headers_seen.append(dict(self.headers.items()))
        if self.path != "/v1/decisions":
            self.send_error(404)
            return
        selector = payload["input"][0]["content"]
        if selector == "auth-failover" and payload["model"] == "gpt-6-luna":
            self._send(
                401,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                        "message": "Incorrect API key provided.",
                    }
                },
            )
            return
        self._send(200, {"model": payload["model"], "answers": _answers(selector), "usage": _USAGE})

    def _send(self, status: int, body: JsonObject) -> None:
        """Write one JSON response."""
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep synthetic HTTP access logs out of test output."""
        del format, args


def _deployment(root: Path, alias: str, wire_model: str) -> tuple[NormalizedGatewayCatalog, Path]:
    """Register one OpenAI decisions deployment priced input-only."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name="openai-loopback",
        provider_model=wire_model,
        exact_model_id="gpt-6-luna-decisions",
        revision=None,
        capabilities=ModelCapabilities(supports_completions=False, supports_embeddings=False),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_decisions=True),
        # OpenAI reports cache reads and writes; the lane prices both at the
        # input rate (the wire bills input only), or their cost stays unknown.
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=_INPUT_RATE,
            cached_input_nano_usd_per_million_tokens=_INPUT_RATE,
            cache_creation_input_nano_usd_per_million_tokens=_INPUT_RATE,
            cache_creation_1h_input_nano_usd_per_million_tokens=_INPUT_RATE,
            output_nano_usd_per_million_tokens=0,
        ),
        pricing_source=None,
        replace=False,
    )
    return normalized, snapshot


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve the shared native driver with one OpenAI decision alias and a failover pool.

    Yields:
        Loopback serving facts for a real gateway and its seeded owning key.
    """
    root = tmp_path_factory.mktemp("native-openai-decisions-root")
    upstream = ThreadingHTTPServer((_HOST, 0), _DecisionsUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_port = upstream.server_address[1]
    manager, raw_key = _configured_gateway(root, base_url=f"http://{_HOST}:{upstream_port}/v1")
    upsert_connection(
        root,
        name="openai-loopback",
        connection=ConnectionConfig(provider="openai", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    normalized, snapshot = _deployment(root, "luna-decisions", "gpt-6-luna")
    manager.activate_direct_alias(
        alias_id="luna-decisions",
        alias_name="luna-decisions",
        revision_id="revision-luna",
        pool_id="luna-decisions",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="luna-decisions")
    _deployment(root, "auth-primary", "gpt-6-luna")
    normalized, _snapshot = _deployment(root, "auth-backup", "gpt-6-luna-backup")
    normalized, snapshot, _changed = upsert_certified_pool(
        root,
        pool_id="luna-auth-failover",
        exact_model_id="gpt-6-luna-decisions",
        deployment_aliases=("auth-primary", "auth-backup"),
        certification=GatewayEquivalenceCertification(
            certification_id="synthetic-openai-decision-equivalence",
            provenance="Both loopback deployments serve these deterministic test answers",
            evidence_sha256="c" * 64,
            certified_at=datetime.now(UTC),
        ),
        expected_catalog_sha256=normalized.identity_sha256(),
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="luna-auth-failover",
        alias_name="luna-auth-failover",
        revision_id="revision-luna-auth-failover",
        pool_id="luna-auth-failover",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="luna-auth-failover")
    SQLiteBudgetStore(manager.database_path).set_limit(
        organization_id=manager.organization_id,
        period=datetime.now(UTC).strftime("%Y-%m"),
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=100_000_000,
    )
    driver = root / "native_openai_decisions_driver.py"
    driver.write_text(_PROFILE_OVERRIDE_SOURCE + "\n\n" + _DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(root),
            "request_timeout_seconds": _TIMEOUT_SECONDS,
            "openai_decisions_loopback_url": f"http://{_HOST}:{upstream_port}/v1/decisions",
        }
    )
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = _PROVIDER_KEY
    stderr_sink = stderr_log.open("wb")
    process = subprocess.Popen(  # noqa: S603 - runs only our generated test driver.
        [sys.executable, str(driver), config],
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
        env=environment,
        text=True,
    )
    try:
        announced_ports: list[int] = []

        def _collect_announcements() -> None:
            """Collect fresh ports if the shared driver must retry a lost bind race."""
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        threading.Thread(target=_collect_announcements, daemon=True).start()
        live_deadline = time.monotonic() + 20.0
        while True:
            if announced_ports:
                port = announced_ports[-1]
                try:
                    models = httpx.get(
                        f"http://{_HOST}:{port}/v1/models",
                        headers={"authorization": f"Bearer {raw_key}"},
                        timeout=1.0,
                    )
                    if models.status_code == 200 and sorted(
                        model["id"] for model in models.json()["data"]
                    ) == ["coding", "luna-auth-failover", "luna-decisions"]:
                        break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, stderr_log.read_text()
            time.sleep(0.05)
        yield _ServingEngine(port=port, raw_key=raw_key, root=root)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=15)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def _post(engine: _ServingEngine, body: JsonObject) -> httpx.Response:
    """Send one public OpenAI-shaped decision request using the seeded owning key."""
    return httpx.post(
        f"{engine.base}/v1/decisions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=body,
        timeout=_TIMEOUT_SECONDS + 3.0,
    )


def _settled(engine: _ServingEngine, request_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row]]:
    """Wait for one request and its attempts to reach a durable terminal."""
    deadline = time.monotonic() + 3.0
    while True:
        with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
            connection.row_factory = sqlite3.Row
            request = connection.execute(
                "SELECT * FROM gateway_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            attempts = connection.execute(
                "SELECT * FROM gateway_attempts WHERE request_id = ? ORDER BY attempt_ordinal",
                (request_id,),
            ).fetchall()
        if (
            request is not None
            and request["terminal_at"] is not None
            and all(attempt["terminal_at"] is not None for attempt in attempts)
        ):
            return request, attempts
        assert time.monotonic() < deadline, request_id
        time.sleep(0.025)


def test_answers_pass_through_with_alias_usage_and_input_only_settlement(
    engine: _ServingEngine,
) -> None:
    """The HTTP route forwards the body, returns OpenAI's shape, and bills input tokens."""
    response = _post(engine, _body())
    assert response.status_code == 200, response.text
    assert response.json() == {
        "model": "luna-decisions",
        "answers": _answers("answer"),
        "usage": _USAGE,
    }
    with _DecisionsUpstream.lock:
        assert _DecisionsUpstream.payloads[-1] == {**_body(), "model": "gpt-6-luna"}
        headers = {key.lower(): value for key, value in _DecisionsUpstream.headers_seen[-1].items()}
    assert headers["authorization"] == f"Bearer {_PROVIDER_KEY}"
    request, [attempt] = _settled(engine, response.headers["x-request-id"])
    assert request["api_surface"] == "decisions"
    assert request["terminal_state"] == "completed"
    assert attempt["state"] == "completed"
    assert attempt["input_tokens"] == 275
    assert attempt["output_tokens"] == 0
    assert attempt["usage_source"] == "observed"
    assert attempt["output_rate"] == 0
    assert attempt["cached_input_tokens"] == 64
    assert attempt["budget_settled_nano_usd"] == 275 * 100


def test_a_declined_question_passes_through_as_a_refusal(engine: _ServingEngine) -> None:
    """One refused question keeps the other answers and settles as completed."""
    response = _post(engine, _body("refusal"))
    assert response.status_code == 200, response.text
    assert response.json()["answers"][2] == {"type": "refusal", "name": "urgency"}
    request, _attempts = _settled(engine, response.headers["x-request-id"])
    assert request["terminal_state"] == "completed"


def test_a_rejected_credential_fails_over_to_the_certified_backup(engine: _ServingEngine) -> None:
    """An OpenAI 401 is a known pre-execution rejection: the backup rung serves it."""
    response = _post(engine, _body("auth-failover", model="luna-auth-failover"))
    assert response.status_code == 200, response.text
    assert response.headers["x-gateway-route-depth"] == "1"
    request, attempts = _settled(engine, response.headers["x-request-id"])
    assert request["terminal_state"] == "completed"
    assert [attempt["state"] for attempt in attempts] == ["failed", "completed"]
    assert attempts[0]["budget_settled_nano_usd"] == 0


def test_an_answer_outside_the_question_fails_closed(engine: _ServingEngine) -> None:
    """A typed choice the request never offered is malformed, never relayed."""
    with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
        before = {row[0] for row in connection.execute("SELECT request_id FROM gateway_requests")}
    response = _post(engine, _body("malformed"))
    assert response.status_code == 502, response.text
    assert "canary" not in response.text
    with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
        [request_id] = {
            row[0] for row in connection.execute("SELECT request_id FROM gateway_requests")
        } - before
    request, [attempt] = _settled(engine, request_id)
    assert request["terminal_state"] == "failed"
    assert attempt["failure_class"] == "malformed_response"

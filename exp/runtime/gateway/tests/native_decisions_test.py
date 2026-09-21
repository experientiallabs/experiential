"""Exercise real native SystemOne dispatch and SQLite settlement on loopback.

Only the TypeSafe-shaped HTTP upstream is synthetic. The shared serving driver
runs Rust admission/dispatch/cancellation through the Python control plane and a
real SQLite authority, attempt ledger, and monthly budget. A driver-local wire
profile replacement downgrades an explicitly trusted loopback HTTPS connection
to HTTP without relaxing TypeSafe's production origin or client validation.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import sqlite3
import struct
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.common.models.gateway_pools import GatewayEquivalenceCertification
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.catalog_authority import (
    upsert_certified_pool,
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.native_messages_test import _DRIVER_SOURCE, _HOST, _ServingEngine

pytest.importorskip("exp_gateway_native")

_TIMEOUT_SECONDS = 8.0
_PROVIDER_KEY = "typesafe-loopback-secret-canary"
_INPUT_RATE = 42_000_000
_COST_NANO_USD = 18_942

_PROFILE_OVERRIDE_SOURCE = textwrap.dedent(
    '''
    """Keep the production TypeSafe client strict; change only this test's wire."""

    import json
    import sys
    from dataclasses import replace
    from urllib.parse import urlsplit

    from exp.runtime.models.providers.base import GatewayWireProfile
    from exp.runtime.models.providers.typesafe import TypeSafeClient

    _original_typesafe_profile = TypeSafeClient.gateway_wire_profile
    _loopback_url = json.loads(sys.argv[1])["typesafe_loopback_url"]


    def _loopback_profile(self: TypeSafeClient) -> GatewayWireProfile:
        """Downgrade only the same-host synthetic endpoint after real construction."""
        profile = _original_typesafe_profile(self)
        official = urlsplit(profile.url)
        loopback = urlsplit(_loopback_url)
        assert official.scheme == "https" and loopback.scheme == "http"
        assert official.hostname == loopback.hostname == "127.0.0.1"
        assert official.port == loopback.port and official.path == loopback.path
        return replace(profile, url=_loopback_url, decisions_url=_loopback_url)


    TypeSafeClient.gateway_wire_profile = _loopback_profile
    '''
).strip()


def _body(selector: str = "answer") -> JsonObject:
    """Build all three question types, preserving structured state and criteria."""
    return {
        "model": "decision",
        "state": {"scenario": selector, "message": "Refund the duplicate invoice", "count": 3},
        "questions": {
            "paid": {"type": "noul", "instructions": "Was payment received?"},
            "department": {
                "type": "choice",
                "instructions": "Choose the department",
                "criteria": {"billing": "Invoices", "technical": "Bugs", "sales": None},
            },
            "quantity": {
                "type": "score",
                "instructions": "Count items",
                "criteria": ["No items", "One or two items", "Three or more items"],
            },
        },
    }


def _answers() -> JsonObject:
    """Return deterministic, nontrivial distributions and their expected score."""
    return {
        "paid": {"type": "noul", "noul": 0.99},
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.8, "technical": 0.15, "sales": 0.05},
        },
        "quantity": {
            "type": "score",
            "score": 1.75,
            "confidence": 0.8,
            "legend": {"0": "No items", "1": "One or two items", "2": "Three or more items"},
            "probabilities": {"0": 0.05, "1": 0.15, "2": 0.8},
        },
    }


class _DecisionsUpstream(BaseHTTPRequestHandler):
    """Serve bounded synthetic answers, recording only loopback test traffic."""

    payloads: list[JsonObject] = []
    headers_seen: list[dict[str, str]] = []
    payloads_lock = threading.Lock()
    stall_started = threading.Event()
    release_stalls = threading.Event()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Answer valid requests, corrupt selected fields, or stall after headers."""
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        with self.payloads_lock:
            self.payloads.append(payload)
            self.headers_seen.append(dict(self.headers.items()))
        if self.path != "/v1/systemone":
            self.send_error(404)
            return
        selector = payload["state"]["scenario"]
        if selector in {"overload", "auth-failover"} and payload["model"] == "jev-latest":
            status = 529 if selector == "overload" else 401
            self.send_error(status, "Synthetic provider rejection")
            return
        answers = _answers()
        body: JsonObject = {
            "model": "jev-1.13.0",
            "answers": answers,
            "usage": {"input_tokens": 451, "output_tokens": 68},
            "provider_private": "do-not-expose-provider-internals",
        }
        if selector == "missing-usage":
            del body["usage"]
        elif selector == "wrong-type":
            answers["paid"] = {"type": "choice", "noul": 0.99}
        elif selector == "wrong-probabilities":
            cast(JsonObject, answers["department"])["probabilities"] = {"billing": 1.0}
        elif selector == "wrong-legend":
            cast(JsonObject, answers["quantity"])["legend"] = {"0": "wrong criteria"}
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        if selector in {"disconnect", "timeout"}:
            self.wfile.flush()
            self.stall_started.set()
            self.release_stalls.wait(timeout=_TIMEOUT_SECONDS + 10.0)
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            # The real native timeout/disconnect closes the provider connection.
            pass

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep synthetic HTTP access logs out of test output."""
        del format, args


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve the shared native driver with separately granted chat/decision aliases.

    Yields:
        Loopback serving facts for a real gateway and its seeded owning key.
    """
    root = tmp_path_factory.mktemp("native-decisions-root")
    with _DecisionsUpstream.payloads_lock:
        _DecisionsUpstream.payloads.clear()
        _DecisionsUpstream.headers_seen.clear()
    _DecisionsUpstream.release_stalls.clear()
    _DecisionsUpstream.stall_started.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _DecisionsUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_port = upstream.server_address[1]
    manager, raw_key = _configured_gateway(root, base_url=f"http://{_HOST}:{upstream_port}/v1")
    upsert_connection(
        root,
        name="typesafe-loopback",
        connection=ConnectionConfig(
            provider="typesafe",
            base_url=f"https://{_HOST}:{upstream_port}/v1",
            trusted_custom_origin=True,
            api_key_env="TEST_PROVIDER_KEY",
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="decision",
        connection_name="typesafe-loopback",
        provider_model="jev-latest",
        exact_model_id="typesafe-jev-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_decisions=True),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=_INPUT_RATE,
            output_nano_usd_per_million_tokens=0,
        ),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="decision",
        alias_name="decision",
        revision_id="revision-decision",
        pool_id="decision",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="decision")
    normalized, _snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="decision-backup",
        connection_name="typesafe-loopback",
        provider_model="jev-backup",
        exact_model_id="typesafe-jev-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_decisions=True),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=_INPUT_RATE,
            output_nano_usd_per_million_tokens=0,
        ),
        pricing_source=None,
        replace=False,
    )
    normalized, snapshot, _changed = upsert_certified_pool(
        root,
        pool_id="decision-failover",
        exact_model_id="typesafe-jev-exact",
        deployment_aliases=("decision", "decision-backup"),
        certification=GatewayEquivalenceCertification(
            certification_id="synthetic-decision-equivalence",
            provenance="Both loopback deployments serve these deterministic test answers",
            evidence_sha256="a" * 64,
            certified_at=datetime.now(UTC),
        ),
        expected_catalog_sha256=normalized.identity_sha256(),
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="decision-failover",
        alias_name="decision-failover",
        revision_id="revision-decision-failover",
        pool_id="decision-failover",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="decision-failover")
    # Separate deployments keep prior malformed/overload circuits out of the
    # auth-rejection test without bypassing production health policy.
    for alias, wire_model in (("auth-primary", "jev-latest"), ("auth-backup", "jev-backup")):
        normalized, _snapshot, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=alias,
            connection_name="typesafe-loopback",
            provider_model=wire_model,
            exact_model_id="typesafe-auth-exact",
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_decisions=True),
            prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=_INPUT_RATE,
                output_nano_usd_per_million_tokens=0,
            ),
            pricing_source=None,
            replace=False,
        )
    normalized, snapshot, _changed = upsert_certified_pool(
        root,
        pool_id="decision-auth-failover",
        exact_model_id="typesafe-auth-exact",
        deployment_aliases=("auth-primary", "auth-backup"),
        certification=GatewayEquivalenceCertification(
            certification_id="synthetic-auth-decision-equivalence",
            provenance="Both loopback deployments serve these deterministic test answers",
            evidence_sha256="b" * 64,
            certified_at=datetime.now(UTC),
        ),
        expected_catalog_sha256=normalized.identity_sha256(),
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="decision-auth-failover",
        alias_name="decision-auth-failover",
        revision_id="revision-decision-auth-failover",
        pool_id="decision-auth-failover",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="decision-auth-failover")
    SQLiteBudgetStore(manager.database_path).set_limit(
        organization_id=manager.organization_id,
        period=datetime.now(UTC).strftime("%Y-%m"),
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=100_000_000,
    )
    driver = root / "native_decisions_driver.py"
    driver.write_text(_PROFILE_OVERRIDE_SOURCE + "\n\n" + _DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(root),
            "request_timeout_seconds": _TIMEOUT_SECONDS,
            "typesafe_loopback_url": f"http://{_HOST}:{upstream_port}/v1/systemone",
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
                    ) == ["coding", "decision", "decision-auth-failover", "decision-failover"]:
                        break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, stderr_log.read_text()
            time.sleep(0.05)
        yield _ServingEngine(port=port, raw_key=raw_key, root=root)
    finally:
        _DecisionsUpstream.release_stalls.set()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=15)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def _post(engine: _ServingEngine, body: JsonObject, **headers: str) -> httpx.Response:
    """Send one public native decision request using the seeded owning key."""
    return httpx.post(
        f"{engine.base}/v1/systemone",
        headers={"authorization": f"Bearer {engine.raw_key}", **headers},
        json=body,
        timeout=_TIMEOUT_SECONDS + 3.0,
    )


def _request_ids(engine: _ServingEngine) -> set[str]:
    """Snapshot accepted IDs directly from the real SQLite request ledger."""
    with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
        return {
            str(row[0]) for row in connection.execute("SELECT request_id FROM gateway_requests")
        }


def _settled(
    engine: _ServingEngine,
    before: set[str],
    *,
    expected: int = 1,
    timeout: float = 3.0,
) -> list[tuple[sqlite3.Row, list[sqlite3.Row]]]:
    """Wait for every newly accepted request and attempt to reach a durable terminal.

    Args:
        engine: Live native server with the owning SQLite root.
        before: IDs accepted before the operation under test.
        expected: Exact number of new accepted requests.
        timeout: Bound on asynchronous disconnect settlement.

    Returns:
        Newly settled request rows paired with their ordered physical attempts.
    """
    deadline = time.monotonic() + timeout
    while True:
        with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = [
                (
                    request,
                    connection.execute(
                        "SELECT * FROM gateway_attempts WHERE request_id = ?"
                        " ORDER BY attempt_ordinal",
                        (request["request_id"],),
                    ).fetchall(),
                )
                for request in connection.execute("SELECT * FROM gateway_requests").fetchall()
                if request["request_id"] not in before
            ]
        if len(rows) == expected and all(
            request["terminal_at"] is not None
            and all(attempt["terminal_at"] is not None for attempt in attempts)
            for request, attempts in rows
        ):
            return rows
        assert time.monotonic() < deadline, [
            (dict(request), [dict(attempt) for attempt in attempts]) for request, attempts in rows
        ]
        time.sleep(0.025)


def _assert_budget_accounted(engine: _ServingEngine) -> None:
    """Reconcile terminal attempt charges, keeping unknown liability held, not spent."""
    with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
        [(held, spent, unknown)] = connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd, unknown_cost_attempts "
            "FROM gateway_monthly_budgets"
        ).fetchall()
        assert unknown == 0  # Every test deployment has a known conservative price bound.
        assert (
            held
            == connection.execute(
                "SELECT COALESCE(SUM(reserved_nano_usd), 0) FROM gateway_attempt_budget_charges "
                "WHERE settled_nano_usd IS NULL"
            ).fetchone()[0]
        )
        assert (
            spent
            == connection.execute(
                "SELECT COALESCE(SUM(settled_nano_usd), 0) FROM gateway_attempt_budget_charges"
            ).fetchone()[0]
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM gateway_attempts WHERE state = 'dispatched'"
        ).fetchone() == (0,)


def _assert_unknown_liability(engine: _ServingEngine, attempt: sqlite3.Row) -> None:
    """A terminal dispatched call with unknown usage retains its precise budget hold."""
    assert attempt["input_tokens"] is None
    assert attempt["output_tokens"] is None
    assert attempt["usage_source"] == "unknown"
    assert attempt["estimated_cost_nano_usd"] is None
    assert attempt["budget_settled_nano_usd"] is None
    assert attempt["budget_reserved_nano_usd"] > 0
    with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
        assert connection.execute(
            "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_attempt_budget_charges "
            "WHERE attempt_id = ?",
            (attempt["attempt_id"],),
        ).fetchall() == [(attempt["budget_reserved_nano_usd"], None)]
    _assert_budget_accounted(engine)


def _provider_calls() -> int:
    """Read the physical upstream dispatch count under its recording lock."""
    with _DecisionsUpstream.payloads_lock:
        return len(_DecisionsUpstream.payloads)


def test_typed_answers_preserve_alias_and_settle_exact_owned_usage(engine: _ServingEngine) -> None:
    """Native answers preserve every question type and bill 451 input tokens exactly."""
    before = _request_ids(engine)
    response = _post(engine, _body(), **{"x-client-request-id": "decision-correlation"})
    assert response.status_code == 200, response.text
    public = response.json()
    assert re.fullmatch(r"decision_[0-9a-f]{32}", public["id"])
    assert public["model"] == "decision"
    assert public["answers"] == _answers()
    assert public["usage"] == {"input_tokens": 451, "output_tokens": 68}
    assert "provider_private" not in public
    assert response.headers["x-gateway-alias"] == "decision"
    assert response.headers["x-gateway-provider"] == "typesafe"
    assert response.headers["x-gateway-route-depth"] == "0"
    assert response.headers["x-client-request-id"] == "decision-correlation"
    with _DecisionsUpstream.payloads_lock:
        assert _DecisionsUpstream.payloads[-1] == {**_body(), "model": "jev-latest"}
        headers = {key.lower(): value for key, value in _DecisionsUpstream.headers_seen[-1].items()}
    assert headers["authorization"] == f"Bearer {_PROVIDER_KEY}"
    assert engine.raw_key not in headers.values()
    [(request, attempts)] = _settled(engine, before)
    assert request["request_id"] == response.headers["x-request-id"]
    assert request["key_id"] == "key-one"
    assert request["identity_id"] == "default"
    assert request["organization_id"] == GatewayManagement(engine.root).organization_id
    assert request["alias_id"] == "decision"
    assert request["api_surface"] == "decisions"
    assert request["terminal_state"] == "completed"
    assert request["caller_operation_sha256"] is None
    assert request["content_retained"] == 0
    [attempt] = attempts
    assert attempt["state"] == "completed"
    assert attempt["failure_class"] is None
    assert attempt["provider"] == "typesafe"
    assert attempt["exact_model_id"] == "typesafe-jev-exact"
    assert attempt["input_tokens"] == 451
    assert attempt["output_tokens"] == 68
    assert attempt["usage_source"] == "observed"
    assert attempt["input_rate"] == _INPUT_RATE
    assert attempt["output_rate"] == 0
    assert attempt["estimated_cost_nano_usd"] == _COST_NANO_USD
    assert attempt["budget_settled_nano_usd"] == _COST_NANO_USD
    assert attempt["budget_reserved_nano_usd"] > _COST_NANO_USD
    assert attempt["content_retained"] == 0
    _assert_budget_accounted(engine)


def test_missing_usage_and_invalid_typed_answers_fail_closed(engine: _ServingEngine) -> None:
    """Unbillable answers and mismatched types/probabilities/legends never escape."""
    cases = (
        ("missing-usage", "decision"),
        ("wrong-type", "decision"),
        ("wrong-probabilities", "decision"),
        ("wrong-legend", "decision"),
        ("wrong-type", "decision-failover"),
    )
    for selector, alias in cases:
        before = _request_ids(engine)
        calls_before = _provider_calls()
        response = _post(engine, {**_body(selector), "model": alias})
        assert response.status_code == 502, response.text
        assert response.json()["error"]["code"] == "all_routes_failed"
        assert "answers" not in response.json()
        assert _provider_calls() == calls_before + 1
        with _DecisionsUpstream.payloads_lock:
            assert _DecisionsUpstream.payloads[-1]["model"] == "jev-latest"
        [(request, attempts)] = _settled(engine, before)
        assert request["alias_id"] == alias
        assert request["terminal_state"] == "failed"
        [attempt] = attempts
        assert attempt["state"] == "failed"
        assert attempt["failure_class"] == "malformed_response"
        assert attempt["route_depth"] == 0
        _assert_unknown_liability(engine, attempt)


def test_authentication_wrong_model_and_chat_misuse_stop_before_dispatch(
    engine: _ServingEngine,
) -> None:
    """Unknown keys are 401; wrong surfaces are explicit model-field 400 refusals."""
    calls_before = _provider_calls()
    before = _request_ids(engine)
    for headers in ({}, {"authorization": "Bearer not-a-key"}):
        response = httpx.post(
            f"{engine.base}/v1/systemone", headers=headers, json=_body(), timeout=3.0
        )
        assert response.status_code == 401, response.text
        assert response.json()["error"]["code"] == "invalid_key"
    assert _request_ids(engine) == before
    wrong_model = _post(engine, {**_body(), "model": "coding"})
    assert wrong_model.status_code == 400, wrong_model.text
    assert wrong_model.json()["error"]["code"] == "unsupported_capability"
    assert wrong_model.json()["error"]["param"] == "model"
    chat = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "decision", "messages": [{"role": "user", "content": "Do not translate"}]},
        timeout=3.0,
    )
    assert chat.status_code == 400, chat.text
    assert chat.json()["error"]["code"] == "unsupported_capability"
    assert chat.json()["error"]["param"] == "model"
    for request, attempts in _settled(engine, before, expected=2):
        assert request["terminal_state"] == "failed"
        assert attempts == []
    assert _provider_calls() == calls_before
    _assert_budget_accounted(engine)


def test_repeated_decisions_are_distinct_and_only_auth_rejection_fails_over(
    engine: _ServingEngine,
) -> None:
    """Repeated calls stay distinct; overload holds liability, auth rejection may fail over."""
    before = _request_ids(engine)
    calls_before = _provider_calls()
    responses = [
        _post(engine, _body("repeat"), **headers)
        for headers in ({}, {}, {"Idempotency-Key": "same-key"}, {"Idempotency-Key": "same-key"})
    ]
    assert all(response.status_code == 200 for response in responses), [
        response.text for response in responses
    ]
    assert len({response.headers["x-request-id"] for response in responses}) == 4
    assert len({response.json()["id"] for response in responses}) == 4
    assert all(
        re.fullmatch(r"decision_[0-9a-f]{32}", response.json()["id"]) for response in responses
    )
    assert _provider_calls() == calls_before + 4
    with _DecisionsUpstream.payloads_lock:
        assert all(
            "idempotency-key" not in {name.lower() for name in headers}
            for headers in _DecisionsUpstream.headers_seen[calls_before:]
        )
    rows = _settled(engine, before, expected=4)
    assert len({request["canonical_request_sha256"] for request, _attempts in rows}) == 1
    for request, attempts in rows:
        assert request["caller_operation_sha256"] is None
        assert request["terminal_state"] == "completed"
        [attempt] = attempts
        assert attempt["state"] == "completed"
        assert attempt["estimated_cost_nano_usd"] == _COST_NANO_USD
    _assert_budget_accounted(engine)

    before = _request_ids(engine)
    calls_before = _provider_calls()
    response = _post(engine, {**_body("overload"), "model": "decision-failover"})
    assert response.status_code == 502, response.text
    assert _provider_calls() == calls_before + 1
    with _DecisionsUpstream.payloads_lock:
        assert _DecisionsUpstream.payloads[-1]["model"] == "jev-latest"
    [(request, attempts)] = _settled(engine, before)
    assert request["terminal_state"] == "failed"
    [attempt] = attempts
    assert attempt["state"] == "failed"
    assert attempt["route_depth"] == 0
    assert attempt["failure_class"] == "provider_internal"
    _assert_unknown_liability(engine, attempt)

    before = _request_ids(engine)
    calls_before = _provider_calls()
    response = _post(engine, {**_body("auth-failover"), "model": "decision-auth-failover"})
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "decision-auth-failover"
    assert response.json()["answers"] == _answers()
    assert response.headers["x-gateway-route-depth"] == "1"
    assert _provider_calls() == calls_before + 2
    with _DecisionsUpstream.payloads_lock:
        assert [payload["model"] for payload in _DecisionsUpstream.payloads[-2:]] == [
            "jev-latest",
            "jev-backup",
        ]
    [(request, attempts)] = _settled(engine, before)
    assert request["terminal_state"] == "completed"
    assert [(attempt["attempt_ordinal"], attempt["route_depth"]) for attempt in attempts] == [
        (0, 0),
        (1, 1),
    ]
    failed, completed = attempts
    assert failed["state"] == "failed"
    assert failed["failure_class"] == "invalid_request"
    assert failed["estimated_cost_nano_usd"] is None
    assert failed["budget_settled_nano_usd"] == 0
    assert failed["usage_source"] == "unknown"
    assert failed["input_tokens"] is None
    assert failed["output_tokens"] is None
    assert completed["state"] == "completed"
    assert completed["estimated_cost_nano_usd"] == _COST_NANO_USD
    _assert_budget_accounted(engine)


def test_disconnect_settles_before_deadline_and_holds_unknown_liability(
    engine: _ServingEngine,
) -> None:
    """A hard socket disconnect exercises native cancellation, not a later timeout."""
    before = _request_ids(engine)
    calls_before = _provider_calls()
    _DecisionsUpstream.stall_started.clear()
    encoded = json.dumps(_body("disconnect")).encode()
    started = time.monotonic()
    with socket.create_connection((_HOST, engine.port), timeout=3.0) as client:
        client.sendall(
            (
                "POST /v1/systemone HTTP/1.1\r\n"
                f"Host: {_HOST}:{engine.port}\r\n"
                f"Authorization: Bearer {engine.raw_key}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(encoded)}\r\n\r\n"
            ).encode()
            + encoded
        )
        assert _DecisionsUpstream.stall_started.wait(timeout=2.0)
        with sqlite3.connect(GatewayManagement(engine.root).database_path) as connection:
            assert (
                connection.execute(
                    "SELECT reserved_nano_usd FROM gateway_monthly_budgets"
                ).fetchone()[0]
                > 0
            )
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    [(request, attempts)] = _settled(engine, before, timeout=4.0)
    assert time.monotonic() - started < _TIMEOUT_SECONDS - 1.0
    assert request["terminal_state"] == "cancelled"
    [attempt] = attempts
    assert attempt["state"] == "cancelled"
    assert _provider_calls() == calls_before + 1
    _assert_unknown_liability(engine, attempt)


def test_timeout_retains_unknown_liability_and_exhausts_budget(engine: _ServingEngine) -> None:
    """A native timeout is terminal, but its retained liability still limits future calls."""
    before = _request_ids(engine)
    calls_before = _provider_calls()
    _DecisionsUpstream.stall_started.clear()
    started = time.monotonic()
    response = _post(engine, _body("timeout"))
    assert response.status_code == 504, response.text
    assert time.monotonic() - started < _TIMEOUT_SECONDS + 3.0
    assert _DecisionsUpstream.stall_started.is_set()
    [(request, attempts)] = _settled(engine, before)
    assert request["terminal_state"] == "failed"
    [attempt] = attempts
    assert attempt["state"] == "failed"
    assert attempt["failure_class"] == "timeout"
    assert _provider_calls() == calls_before + 1
    _assert_unknown_liability(engine, attempt)

    manager = GatewayManagement(engine.root)
    with sqlite3.connect(manager.database_path) as connection:
        [(limit, held, spent)] = connection.execute(
            "SELECT limit_nano_usd, reserved_nano_usd, settled_nano_usd "
            "FROM gateway_monthly_budgets"
        ).fetchall()
    assert held >= attempt["budget_reserved_nano_usd"] > 1
    budget = SQLiteBudgetStore(manager.database_path)
    period = datetime.now(UTC).strftime("%Y-%m")
    budget.set_limit(
        organization_id=manager.organization_id,
        period=period,
        scope=BudgetScope(kind=BudgetScopeKind.TEAM),
        limit_nano_usd=spent + held + 1,
        replace=True,
    )
    try:
        before = _request_ids(engine)
        calls_before = _provider_calls()
        blocked = _post(engine, _body("timeout"))
        assert blocked.status_code == 429, blocked.text
        assert blocked.json()["error"]["code"] == "insufficient_quota"
        assert _provider_calls() == calls_before
        [(request, attempts)] = _settled(engine, before)
        assert request["terminal_state"] == "failed"
        assert attempts == []
        with sqlite3.connect(manager.database_path) as connection:
            assert connection.execute(
                "SELECT reserved_nano_usd, settled_nano_usd FROM gateway_monthly_budgets"
            ).fetchall() == [(held, spent)]
        _assert_budget_accounted(engine)
    finally:
        budget.set_limit(
            organization_id=manager.organization_id,
            period=period,
            scope=BudgetScope(kind=BudgetScopeKind.TEAM),
            limit_nano_usd=limit,
            replace=True,
        )

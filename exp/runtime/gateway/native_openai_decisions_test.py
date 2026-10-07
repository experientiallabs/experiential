"""SQLite-backed admission tests for OpenAI-shaped decisions, without provider calls."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_decisions_test import _control_plane as _systemone_control_plane
from exp.runtime.gateway.openai_decisions_contracts import OpenAIDecisionRequest

_PRICES = GatewayTokenPrices(
    input_nano_usd_per_million_tokens=100_000_000,
    output_nano_usd_per_million_tokens=0,
)


def _control_plane(
    root: Path,
    *,
    supports_decisions: bool = True,
    prices: GatewayTokenPrices = _PRICES,
) -> tuple[NativeControlPlane, str]:
    """Create real authority and ledger state with one direct OpenAI decisions lane."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="openai-main",
        connection=ConnectionConfig(provider="openai", api_key_env="TEST_OPENAI_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="luna-decisions",
        connection_name="openai-main",
        provider_model="gpt-6-luna",
        exact_model_id="gpt-6-luna-decisions",
        revision=None,
        capabilities=ModelCapabilities(supports_completions=False, supports_embeddings=False),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_decisions=supports_decisions),
        prices=prices,
        pricing_source="test-fixture",
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="luna-decisions",
        alias_name="luna-decisions",
        revision_id="revision-one",
        pool_id="luna-decisions",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="luna-decisions")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    components = load_gateway_components(
        root, environment={"TEST_OPENAI_KEY": "openai-secret-canary"}
    )
    return NativeControlPlane(components), issued.raw_key


def _body(model: str = "luna-decisions") -> JsonObject:
    """Return every OpenAI question type, including content-leak canaries."""
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "private-input-canary"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
                ],
            }
        ],
        "questions": [
            {
                "type": "choice",
                "name": "intent",
                "instructions": "private-instructions-canary",
                "choices": [
                    {"value": "refund", "description": "wants money back"},
                    {"value": "support", "description": "needs help"},
                ],
            },
            {"type": "predicate", "name": "angry", "instructions": "Is the customer angry?"},
            {
                "type": "score",
                "name": "urgency",
                "instructions": "How urgent?",
                "levels": [
                    {"label": "low", "description": "can wait"},
                    {"label": "high", "description": "now"},
                ],
            },
        ],
    }


def _admit(control: NativeControlPlane, raw_key: str, body: JsonObject | None = None) -> JsonObject:
    """Call only the Python admission seam, with no provider HTTP work."""
    return json.loads(
        control.admit_openai_decisions(
            json.dumps({"raw_key": raw_key, "body": json.dumps(_body() if body is None else body)})
        )
    )


def _rows(control: NativeControlPlane) -> list[tuple[str, str | None]]:
    """Read durable API surfaces and terminal states, not content-bearing traffic."""
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        return connection.execute(
            "select api_surface, terminal_state from gateway_requests order by rowid"
        ).fetchall()


def _public_error(error: NativeBridgeError) -> JsonObject:
    """Decode one sanitized boundary error."""
    return json.loads(error.public_error_json)


def test_admission_builds_the_openai_decisions_wire_with_input_only_reservation(
    tmp_path: Path,
) -> None:
    """The admitted wire targets OpenAI's own /decisions endpoint on the decisions surface."""
    control, raw_key = _control_plane(tmp_path)
    admitted = _admit(control, raw_key)
    route = admitted["route"]
    assert isinstance(route, list) and len(route) == 1
    wire = route[0]
    assert isinstance(wire, dict)
    assert wire["dialect"] == "openai_responses"
    assert wire["url"] == "https://api.openai.com/v1/decisions"
    assert wire["headers"]["Authorization"] == "Bearer openai-secret-canary"
    expected = _body()
    expected["model"] = "gpt-6-luna"
    assert wire["upstream_payload"] == expected
    assert admitted["wire"] == "openai"
    assert admitted["openai_questions"] == expected["questions"]
    assert "questions" not in admitted
    assert admitted["maximum_same_deployment_attempts"] == 1
    assert _rows(control) == [("decisions", None)]
    entry = control._accounting.entry(str(admitted["request_id"]))  # noqa: SLF001
    assert entry is not None and isinstance(entry.request, OpenAIDecisionRequest)
    assert entry.request.input_token_reservation > 2048
    assert entry.request.output_token_reservation == 0
    assert raw_key not in json.dumps(admitted)


def test_settlement_bills_reported_input_tokens(tmp_path: Path) -> None:
    """A completed OpenAI decision settles on the provider's input-only usage."""
    control, raw_key = _control_plane(tmp_path)
    admitted = _admit(control, raw_key)
    started = json.loads(
        control.start_attempt(
            json.dumps({"request_id": admitted["request_id"], "attempt_ordinal": 0})
        )
    )
    assert started["route_depth"] == 0
    settled = control.settle(
        json.dumps(
            {
                "request_id": admitted["request_id"],
                "attempt_id": started["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 275, "output_tokens": 0},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    assert _rows(control) == [("decisions", "completed")]
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["input_tokens"] == 275
    assert report["totals"]["output_tokens"] == 0


def test_systemone_lane_does_not_serve_the_openai_wire(tmp_path: Path) -> None:
    """An OpenAI-shaped request never reaches a TypeSafe SystemOne rung."""
    control, raw_key = _systemone_control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as refused:
        _admit(control, raw_key, _body(model="decisions"))
    error = _public_error(refused.value)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert _rows(control) == [("decisions", "failed")]


def test_undeclared_or_output_priced_lanes_are_refused(tmp_path: Path) -> None:
    """Decisions support must be declared, and the lane must price output at zero."""
    control, raw_key = _control_plane(tmp_path / "undeclared", supports_decisions=False)
    with pytest.raises(NativeBridgeError) as undeclared:
        _admit(control, raw_key)
    assert _public_error(undeclared.value)["code"] == "unsupported_capability"
    priced = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=100_000_000,
        output_nano_usd_per_million_tokens=500_000_000,
    )
    control, raw_key = _control_plane(tmp_path / "priced", prices=priced)
    with pytest.raises(NativeBridgeError) as unpriced:
        _admit(control, raw_key)
    assert _public_error(unpriced.value)["code"] == "model_unavailable"


def test_invalid_bodies_are_refused_before_acceptance_without_echoing_content(
    tmp_path: Path,
) -> None:
    """Malformed questions, remote images, and extra fields never accept or leak."""
    control, raw_key = _control_plane(tmp_path)
    duplicate = _body()
    duplicate["questions"] = [
        {"type": "predicate", "name": "private-name-canary", "instructions": "a"},
        {"type": "predicate", "name": "private-name-canary", "instructions": "b"},
    ]
    remote = _body()
    remote["input"] = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": "https://example.invalid/a.png"}],
        }
    ]
    streaming = {**_body(), "stream": True}
    for body in (duplicate, remote, streaming):
        with pytest.raises(NativeBridgeError) as refused:
            _admit(control, raw_key, body)
        public = refused.value.public_error_json
        assert _public_error(refused.value)["status_code"] == 400
        assert "canary" not in public
    assert _rows(control) == []

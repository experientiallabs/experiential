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
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
)
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
    provider: str = "openai",
    supports_image_input: bool = True,
) -> tuple[NativeControlPlane, str]:
    """Create real authority and ledger state with one decisions lane.

    ``provider`` is ``openai`` (direct OpenAI, the body forwards verbatim) or
    ``gemini`` (Google's Decisions API, the body is translated).
    """
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="openai-main",
        connection=ConnectionConfig(provider=provider, api_key_env="TEST_OPENAI_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="luna-decisions",
        connection_name="openai-main",
        provider_model="gpt-6-luna" if provider == "openai" else "gemini-decision-1",
        exact_model_id="gpt-6-luna-decisions",
        revision=None,
        capabilities=ModelCapabilities(supports_completions=False, supports_embeddings=False),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_decisions=supports_decisions, supports_image_input=supports_image_input
        ),
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


def test_a_gemini_lane_is_admitted_on_googles_decisions_wire_with_a_translated_body(
    tmp_path: Path,
) -> None:
    """The same OpenAI-shaped request reaches a Gemini lane as a Google Decisions API body."""
    control, raw_key = _control_plane(tmp_path, provider="gemini")
    body = {**_body(), "safety_identifier": "user-123"}
    admitted = _admit(control, raw_key, body)
    route = admitted["route"]
    assert isinstance(route, list) and len(route) == 1
    wire = route[0]
    assert isinstance(wire, dict)
    assert wire["dialect"] == "gemini_generate_content"
    assert wire["url"] == "https://generativelanguage.googleapis.com/v1beta/decisions"
    assert wire["headers"]["x-goog-api-key"] == "openai-secret-canary"
    assert "Authorization" not in wire["headers"]
    payload = wire["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gemini-decision-1"
    assert payload["input"] == [
        {"type": "text", "text": "private-input-canary"},
        {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"},
    ]
    assert set(payload["questions"]) == {"q0", "q1", "q2"}
    assert payload["questions"]["q0"] == {
        "type": "choice",
        "instructions": "private-instructions-canary",
        "criteria": {"refund": "wants money back", "support": "needs help"},
    }
    assert payload["questions"]["q1"] == {"type": "noul", "instructions": "Is the customer angry?"}
    assert payload["questions"]["q2"]["criteria"] == ["low: can wait", "high: now"]
    assert "safety_identifier" not in json.dumps(payload)
    assert admitted["wire"] == "google"
    assert admitted["ignored_parameters"] == ["safety_identifier->dropped(google_decisions)"]
    expected = _body()
    assert admitted["openai_questions"] == expected["questions"]
    assert _rows(control) == [("decisions", None)]
    assert raw_key not in json.dumps(admitted)


def test_an_untranslatable_request_is_refused_on_a_gemini_lane_without_echoing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A choice value Google cannot key is a typed 400 after accept, never a dispatch."""
    control, raw_key = _control_plane(tmp_path, provider="gemini")
    recorded = _recorded_failures(control, monkeypatch)
    body = _body()
    body["questions"] = [
        {
            "type": "choice",
            "instructions": "i",
            "choices": [{"value": " private-value-canary"}, {"value": "other"}],
        }
    ]
    with pytest.raises(NativeBridgeError) as refused:
        _admit(control, raw_key, body)
    error = _public_error(refused.value)
    assert error["status_code"] == 400
    assert error["param"] == "questions"
    assert "128 characters" in str(error["message"])
    assert "canary" not in refused.value.public_error_json
    assert [failure.safe_message for failure in recorded] == [
        "the request cannot be expressed on the model's Google Decisions API wire"
    ]
    assert _rows(control) == [("decisions", "failed")]


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_an_image_request_needs_a_rung_that_declares_image_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """A text-only decision rung never receives an image block; the refusal is typed."""
    control, raw_key = _control_plane(tmp_path, provider=provider, supports_image_input=False)
    recorded = _recorded_failures(control, monkeypatch)
    with pytest.raises(NativeBridgeError) as refused:
        _admit(control, raw_key)
    error = _public_error(refused.value)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "input"
    assert "canary" not in refused.value.public_error_json
    assert [failure.safe_message for failure in recorded] == [
        "no decision deployment accepts images"
    ]
    text_only = _body()
    text_only["input"] = "private-input-canary"
    admitted = _admit(control, raw_key, text_only)
    assert admitted["wire"] == ("openai" if provider == "openai" else "google")


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


def _recorded_failures(
    control: NativeControlPlane, monkeypatch: pytest.MonkeyPatch
) -> list[GatewayFailure]:
    """Record the content-free failures admission settles accepted requests with."""
    recorded: list[GatewayFailure] = []
    accounting = control._accounting  # noqa: SLF001
    finish = accounting.finish_request_quietly

    def _spy(authorization: AuthorizationSnapshot, failure: GatewayFailure) -> None:
        recorded.append(failure)
        finish(authorization, failure)

    monkeypatch.setattr(accounting, "finish_request_quietly", _spy)
    return recorded


def test_systemone_lane_does_not_serve_the_openai_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OpenAI-shaped request never reaches a SystemOne rung; the refusal points there."""
    control, raw_key = _systemone_control_plane(tmp_path)
    recorded = _recorded_failures(control, monkeypatch)
    with pytest.raises(NativeBridgeError) as refused:
        _admit(control, raw_key, _body(model="decisions"))
    error = _public_error(refused.value)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "model"
    assert error["message"] == (
        "The model 'decisions' supports native decisions, not the OpenAI Decisions API. "
        "POST /v1/systemone on this same API host with your same API key "
        "(Authorization: Bearer) and the same model id 'decisions', using model, state "
        "and questions. Key questions by id; each needs type (noul, choice or score) "
        "and instructions (choice also needs a criteria object; score a criteria "
        "array). Read answers[id]; do not send input or a questions list."
    )
    assert [failure.safe_message for failure in recorded] == [
        "the model alias serves native decisions on /v1/systemone, not the OpenAI Decisions API"
    ]
    assert recorded[0].failure_class is GatewayFailureClass.UNSUPPORTED_CAPABILITY
    assert _rows(control) == [("decisions", "failed")]


def test_a_lane_serving_neither_decisions_wire_keeps_the_choose_another_model_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undeclared SystemOne support is not a SystemOne hint: the generic copy stays."""
    control, raw_key = _systemone_control_plane(tmp_path, supports_decisions=False)
    recorded = _recorded_failures(control, monkeypatch)
    with pytest.raises(NativeBridgeError) as refused:
        _admit(control, raw_key, _body(model="decisions"))
    error = _public_error(refused.value)
    assert error["code"] == "unsupported_capability"
    assert error["message"] == (
        "The model 'decisions' does not serve the OpenAI Decisions API. Choose a "
        "decision model served by OpenAI from GET /v1/models and resend the request "
        "to /v1/decisions."
    )
    assert "/v1/systemone" not in json.dumps(error)
    assert [failure.safe_message for failure in recorded] == [
        "the model alias does not serve the OpenAI Decisions API"
    ]


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

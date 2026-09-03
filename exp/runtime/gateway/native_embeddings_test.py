"""Tests for the native embeddings admission boundary."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_bridge import NativeControlPlane


def _control_plane(root: Path, *, embeddings: bool) -> tuple[NativeControlPlane, str]:
    """Seed the ``coding`` alias as an embeddings or chat model and load the plane."""
    capabilities = ModelCapabilities(supports_embeddings=True) if embeddings else None
    _manager, raw_key = _configured_gateway(root, capabilities=capabilities)
    components = load_gateway_components(
        root, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
    )
    return NativeControlPlane(components), raw_key


def _admit(control: NativeControlPlane, raw_key: str, body: JsonObject) -> JsonObject:
    """Run one embeddings admission and decode its JSON answer."""
    argument = json.dumps({"raw_key": raw_key, "body": json.dumps(body)})
    return json.loads(control.admit_embeddings(argument))


def _public_error(exc: NativeBridgeError) -> JsonObject:
    """Decode the OpenAI-shaped public error carried across the boundary."""
    return json.loads(exc.public_error_json)


def _request_row(control: NativeControlPlane, request_id: str) -> tuple[str, str | None]:
    """Read the durable surface and terminal state of one request row."""
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        row = connection.execute(
            "select api_surface, terminal_state from gateway_requests where request_id = ?",
            (request_id,),
        ).fetchone()
    assert row is not None
    return (str(row[0]), None if row[1] is None else str(row[1]))


def test_admit_builds_the_openai_embeddings_wire_and_settles_input_only(tmp_path: Path) -> None:
    """Admission returns one ``/embeddings`` payload per rung; settlement bills input tokens."""
    control, raw_key = _control_plane(tmp_path, embeddings=True)
    admission = _admit(
        control,
        raw_key,
        {
            "model": "coding",
            "input": ["alpha beta", "gamma"],
            "dimensions": 3,
            "encoding_format": "float",
            "user": "tenant-7",
        },
    )
    assert admission["input_count"] == 2
    assert admission["maximum_total_attempts"] == 8
    assert "stream" not in admission
    route = admission["route"]
    assert isinstance(route, list) and len(route) == 1
    wire = route[0]
    assert wire["dialect"] == "openai_compatible"
    assert wire["url"] == "http://127.0.0.1:9/v1/embeddings"
    assert wire["headers"]["Authorization"] == "Bearer provider-secret-canary"
    # `user` is metadata-only: the attribution label at accept, never on the wire.
    assert wire["upstream_payload"] == {
        "model": "provider-model-exact",
        "input": ["alpha beta", "gamma"],
        "dimensions": 3,
        "encoding_format": "float",
    }
    assert wire["upstream_body"] is None
    request_id = str(admission["request_id"])
    assert _request_row(control, request_id) == ("embeddings", None)

    started = json.loads(
        control.start_attempt(
            json.dumps({"request_id": admission["request_id"], "attempt_ordinal": 0})
        )
    )
    assert started["route_depth"] == 0
    settled = control.settle(
        json.dumps(
            {
                "request_id": admission["request_id"],
                "attempt_id": started["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 0},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    assert _request_row(control, request_id) == ("embeddings", "completed")
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["input_tokens"] == 3
    assert report["totals"]["output_tokens"] == 0


def test_admit_rejects_a_chat_alias_with_a_model_field_error_and_finishes_the_request(
    tmp_path: Path,
) -> None:
    """An alias whose rungs never claim embeddings fails closed on ``model``, accounted."""
    control, raw_key = _control_plane(tmp_path, embeddings=False)
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, {"model": "coding", "input": "hello"})
    error = _public_error(raised.value)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "model"
    assert "does not serve embeddings" in str(error["message"])
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["input_tokens"] == 0


def test_admit_rejects_protocol_failures_before_any_ledger_write(tmp_path: Path) -> None:
    """A body the shared decoder rejects never accepts a request."""
    control, raw_key = _control_plane(tmp_path, embeddings=True)
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, {"model": "coding"})
    error = _public_error(raised.value)
    assert error["status_code"] == 400
    assert error["param"] == "input"
    with pytest.raises(NativeBridgeError) as unknown:
        _admit(control, "not-a-key", {"model": "coding", "input": "hello"})
    assert _public_error(unknown.value)["status_code"] == 401
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 0

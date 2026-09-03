"""Tests for the native image-generation admission boundary."""

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


def _control_plane(root: Path, *, images: bool) -> tuple[NativeControlPlane, str]:
    """Seed the ``coding`` alias as an image model or a chat model and load the plane."""
    capabilities = ModelCapabilities(supports_image_generation=True) if images else None
    _manager, raw_key = _configured_gateway(root, capabilities=capabilities)
    components = load_gateway_components(
        root, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
    )
    return NativeControlPlane(components), raw_key


def _admit(control: NativeControlPlane, raw_key: str, body: JsonObject) -> JsonObject:
    argument = json.dumps({"raw_key": raw_key, "body": json.dumps(body)})
    return json.loads(control.admit_images(argument))


def _public_error(exc: NativeBridgeError) -> JsonObject:
    return json.loads(exc.public_error_json)


def _request_row(control: NativeControlPlane, request_id: str) -> tuple[str, str | None]:
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        row = connection.execute(
            "select api_surface, terminal_state from gateway_requests where request_id = ?",
            (request_id,),
        ).fetchone()
    assert row is not None
    return (str(row[0]), None if row[1] is None else str(row[1]))


def test_admit_builds_the_openai_images_wire_and_settles_token_usage(tmp_path: Path) -> None:
    """Admission returns one ``/images/generations`` payload per rung; settle bills both legs."""
    control, raw_key = _control_plane(tmp_path, images=True)
    admission = _admit(
        control,
        raw_key,
        {
            "model": "coding",
            "prompt": "a cat",
            "n": 2,
            "size": "1024x1024",
            "quality": "low",
            "user": "tenant-7",
        },
    )
    assert admission["image_count"] == 2
    route = admission["route"]
    assert isinstance(route, list) and len(route) == 1
    wire = route[0]
    assert wire["url"] == "http://127.0.0.1:9/v1/images/generations"
    assert wire["headers"]["Authorization"] == "Bearer provider-secret-canary"
    # `user` is metadata-only: recorded as the attribution label, never on the wire.
    assert wire["upstream_payload"] == {
        "model": "provider-model-exact",
        "prompt": "a cat",
        "n": 2,
        "size": "1024x1024",
        "quality": "low",
    }
    request_id = str(admission["request_id"])
    assert _request_row(control, request_id) == ("images", None)
    started = json.loads(
        control.start_attempt(json.dumps({"request_id": request_id, "attempt_ordinal": 0}))
    )
    assert (
        control.settle(
            json.dumps(
                {
                    "request_id": request_id,
                    "attempt_id": started["attempt_id"],
                    "outcome": "completed",
                    "usage": {"input_tokens": 11, "output_tokens": 544},
                    "tool_names": [],
                    "failure": None,
                }
            )
        )
        == "{}"
    )
    assert _request_row(control, request_id) == ("images", "completed")
    report = json.loads(control.usage_json("{}"))
    assert report["totals"]["requests"] == 1
    assert report["totals"]["input_tokens"] == 11
    assert report["totals"]["output_tokens"] == 544


def test_admit_rejects_a_chat_alias_on_the_model_field(tmp_path: Path) -> None:
    """An alias whose rungs never claim image generation fails closed, accounted."""
    control, raw_key = _control_plane(tmp_path, images=False)
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, {"model": "coding", "prompt": "a cat"})
    error = _public_error(raised.value)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "model"
    assert "does not generate images" in str(error["message"])
    assert json.loads(control.usage_json("{}"))["totals"]["requests"] == 1


def test_admit_rejects_protocol_failures_before_any_ledger_write(tmp_path: Path) -> None:
    """A missing prompt, a streaming request, and an unknown key never accept a request."""
    control, raw_key = _control_plane(tmp_path, images=True)
    with pytest.raises(NativeBridgeError) as missing:
        _admit(control, raw_key, {"model": "coding"})
    assert _public_error(missing.value)["status_code"] == 400
    with pytest.raises(NativeBridgeError) as streaming:
        _admit(control, raw_key, {"model": "coding", "prompt": "a cat", "stream": True})
    assert _public_error(streaming.value)["status_code"] == 400
    with pytest.raises(NativeBridgeError) as unknown:
        _admit(control, "not-a-key", {"model": "coding", "prompt": "a cat"})
    assert _public_error(unknown.value)["status_code"] == 401
    assert json.loads(control.usage_json("{}"))["totals"]["requests"] == 0

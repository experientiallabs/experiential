"""Tests for the native image-generation admission boundary."""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.common.models.catalog_prices import BilledUnitKind, GatewayTokenPrices, GatewayUnitPrices
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


def test_admit_rejects_an_image_emitting_chat_lane_without_the_images_claim(
    tmp_path: Path,
) -> None:
    """`emits_images` never admits /v1/images: the Images-API claim is the only key.

    2026-09-15: projecting `supports_image_generation` onto the text+image
    chat lanes admitted `/v1/images/generations` on OpenRouter, whose
    OpenAI-compatible profile carries an `images_url` unconditionally, and
    dispatched to a URL that does not exist. The chat-side fact lives under
    its own key and this route ignores it.
    """
    _manager, raw_key = _configured_gateway(
        tmp_path, capabilities=ModelCapabilities(emits_images=True)
    )
    control = NativeControlPlane(
        load_gateway_components(
            tmp_path, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
        )
    )
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, {"model": "coding", "prompt": "a cat"})
    error = _public_error(raised.value)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_capability"
    assert "does not generate images" in str(error["message"])


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


def test_admit_attributes_the_calling_app(tmp_path: Path) -> None:
    """The forwarded User-Agent classifies a images caller onto its durable request."""
    control, raw_key = _control_plane(tmp_path, images=True)
    argument = {
        "raw_key": raw_key,
        "body": json.dumps({"model": "coding", "prompt": "a cat"}),
        "user_agent": "claude-cli/2.1.278 (external, cli)",
    }
    control.admit_images(json.dumps(argument))
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        rows = connection.execute("select client_app, user_agent from gateway_requests").fetchall()
    assert rows == [("claude_code", "claude-cli/2.1.278 (external, cli)")]


def test_admit_drops_b64_json_for_a_gpt_image_model(tmp_path: Path) -> None:
    """``chatgpt-image-latest`` answers base64 and refuses ``response_format``."""
    _manager, raw_key = _configured_gateway(
        tmp_path,
        capabilities=ModelCapabilities(supports_image_generation=True),
        provider_model="chatgpt-image-latest",
    )
    control = NativeControlPlane(
        load_gateway_components(
            tmp_path, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
        )
    )
    admission = _admit(
        control,
        raw_key,
        {"model": "coding", "prompt": "a cat", "response_format": "b64_json"},
    )
    route = admission["route"]
    assert isinstance(route, list) and len(route) == 1
    assert route[0]["upstream_payload"] == {
        "model": "chatgpt-image-latest",
        "prompt": "a cat",
        "n": 1,
    }


def test_admit_refuses_url_on_a_gpt_image_model_naming_the_field(tmp_path: Path) -> None:
    """No rung can answer ``url``: a typed 400 on ``response_format``, accounted, no dispatch."""
    _manager, raw_key = _configured_gateway(
        tmp_path,
        capabilities=ModelCapabilities(supports_image_generation=True),
        provider_model="gpt-image-1",
    )
    control = NativeControlPlane(
        load_gateway_components(
            tmp_path, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
        )
    )
    with pytest.raises(NativeBridgeError) as raised:
        _admit(control, raw_key, {"model": "coding", "prompt": "a cat", "response_format": "url"})
    error = _public_error(raised.value)
    assert error["status_code"] == 400
    assert error["code"] == "unsupported_value"
    assert error["param"] == "response_format"
    assert "base64 images only" in str(error["message"])
    assert json.loads(control.usage_json("{}"))["totals"]["requests"] == 1


_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")

# gpt-image-2.5-sunburst's published rates: text input $5/M, image input $8/M,
# image output $30/M.
_GPT_IMAGE_PRICES = GatewayTokenPrices(
    input_nano_usd_per_million_tokens=5_000_000_000,
    output_nano_usd_per_million_tokens=30_000_000_000,
    image_input_nano_usd_per_million_tokens=8_000_000_000,
)

# grok-imagine-image-2.0 as measured: $0.04 per generated image, $0.06 per
# edited image, $0.01 per source image an edit reads.
_GROK_IMAGINE_UNITS = GatewayUnitPrices(
    kind=BilledUnitKind.IMAGE,
    rates={"": 40_000_000, "edit": 60_000_000},
    input_rates={"": 10_000_000},
)


def _edit_plane(
    root: Path,
    *,
    provider: str = "openai",
    provider_model: str = "gpt-image-2.5-sunburst",
    base_url: str | None = None,
    prices: GatewayTokenPrices = _GPT_IMAGE_PRICES,
) -> tuple[NativeControlPlane, str]:
    """Seed one image rung that may edit and load the plane."""
    _manager, raw_key = _configured_gateway(
        root,
        capabilities=ModelCapabilities(supports_image_generation=True),
        provider=provider,
        provider_model=provider_model,
        base_url=base_url,
        prices=prices,
    )
    components = load_gateway_components(
        root, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
    )
    return NativeControlPlane(components), raw_key


def _admit_edit(
    control: NativeControlPlane, raw_key: str, fields: JsonObject, files: JsonObject
) -> JsonObject:
    """Admit one parsed edit upload naming the ``coding`` alias."""
    upload = {"fields": {"model": "coding", **fields}, "files": files}
    return json.loads(control.admit_image_edits(json.dumps({"raw_key": raw_key, "upload": upload})))


def _edit_error(
    control: NativeControlPlane, raw_key: str, fields: JsonObject, files: JsonObject
) -> JsonObject:
    """Admit one edit upload that must fail and return its public error."""
    with pytest.raises(NativeBridgeError) as raised:
        _admit_edit(control, raw_key, fields, files)
    return _public_error(raised.value)


def test_a_url_edit_on_a_gpt_image_model_is_refused_naming_response_format(
    tmp_path: Path,
) -> None:
    """A GPT image edit answers base64 only, so ``url`` is the field at fault."""
    control, raw_key = _edit_plane(tmp_path)
    error = _edit_error(
        control, raw_key, {"prompt": "a cat", "response_format": "url"}, {"image": _PNG}
    )
    assert error["status_code"] == 400
    assert error["param"] == "response_format"
    assert "base64 images only" in str(error["message"])


def test_a_gpt_image_edit_admits_the_multipart_wire_and_settles_image_tokens(
    tmp_path: Path,
) -> None:
    """The rung's edit endpoint, text fields, and multipart shape reach the data plane."""
    control, raw_key = _edit_plane(tmp_path)
    admission = _admit_edit(
        control,
        raw_key,
        {
            "prompt": "watercolor",
            "n": "2",
            "size": "1024x1024",
            "quality": "low",
            "input_fidelity": "high",
            "moderation": "low",
            "response_format": "b64_json",
        },
        {"image[]": [_PNG, _PNG], "mask": _PNG},
    )
    route = admission["route"]
    assert isinstance(route, list) and len(route) == 1
    assert route[0]["url"] == "https://api.openai.com/v1/images/edits"
    assert route[0]["upstream_payload"] == {
        "model": "gpt-image-2.5-sunburst",
        "n": 2,
        "prompt": "watercolor",
        "size": "1024x1024",
        "quality": "low",
        "input_fidelity": "high",
        "moderation": "low",
    }
    assert admission["wire_shapes"] == [{"shape": "openai", "multipart": True}]
    assert admission["unit_billing"] == [None]
    request_id = str(admission["request_id"])
    started = json.loads(
        control.start_attempt(json.dumps({"request_id": request_id, "attempt_ordinal": 0}))
    )
    settled = control.settle(
        json.dumps(
            {
                "request_id": request_id,
                "attempt_id": started["attempt_id"],
                "outcome": "completed",
                "usage": {
                    "input_tokens": 1554,
                    "image_input_tokens": 1536,
                    "output_tokens": 392,
                },
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    assert _request_row(control, request_id) == ("images", "completed")
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        row = connection.execute(
            "select estimated_cost_nano_usd from gateway_attempts where request_id = ?",
            (request_id,),
        ).fetchone()
    # Image input at $8/M, text input at $5/M, image output at $30/M.
    assert row == ((1536 * 8_000_000_000 + 18 * 5_000_000_000 + 392 * 30_000_000_000) // 1_000_000,)


def test_a_token_edit_rung_without_an_image_input_rate_is_not_admitted(
    tmp_path: Path,
) -> None:
    """Source images are never priced at the text rate: the rung is excluded."""
    control, raw_key = _edit_plane(
        tmp_path,
        prices=_GPT_IMAGE_PRICES.model_copy(
            update={"image_input_nano_usd_per_million_tokens": None}
        ),
    )
    error = _edit_error(control, raw_key, {"prompt": "watercolor"}, {"image": _PNG})
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "model"
    assert "image-input price" in str(error["message"])
    # The same rung still generates.
    assert _admit(control, raw_key, {"model": "coding", "prompt": "a cat"})["route"]


def test_a_style_image_on_a_gpt_image_edit_is_refused_by_name(tmp_path: Path) -> None:
    """A Stability-only part is refused on the OpenAI edit wire, never dropped."""
    control, raw_key = _edit_plane(tmp_path)
    error = _edit_error(
        control, raw_key, {"prompt": "watercolor"}, {"image": _PNG, "style_image": _PNG}
    )
    assert error["param"] == "style_image"


def test_a_non_gpt_openai_compatible_image_rung_does_not_edit(tmp_path: Path) -> None:
    """Only GPT image, Grok Imagine and Stability rungs edit images."""
    control, raw_key = _edit_plane(
        tmp_path, provider="openai-compatible", base_url="http://127.0.0.1:9/v1"
    )
    error = _edit_error(control, raw_key, {"prompt": "watercolor"}, {"image": _PNG})
    assert error["param"] == "model"
    assert "does not edit images" in str(error["message"])


def test_an_xai_edit_converts_the_upload_to_json_and_bills_both_legs(tmp_path: Path) -> None:
    """xAI gets data URIs as JSON; the attempt bills edited and source images."""
    control, raw_key = _edit_plane(
        tmp_path,
        provider="openai-compatible",
        provider_model="grok-imagine-image-2.0",
        base_url="https://api.x.ai/v1",
        prices=GatewayTokenPrices(units=_GROK_IMAGINE_UNITS),
    )
    admission = _admit_edit(
        control, raw_key, {"prompt": "combine", "n": "2"}, {"image[]": [_PNG, _PNG, _PNG]}
    )
    route = admission["route"]
    assert isinstance(route, list) and len(route) == 1
    assert route[0]["url"] == "https://api.x.ai/v1/images/edits"
    payload = route[0]["upstream_payload"]
    assert payload["response_format"] == "b64_json"
    assert len(payload["images"]) == 3
    assert payload["images"][0] == {"url": f"data:image/png;base64,{_PNG}", "type": "image_url"}
    assert admission["wire_shapes"] == [{"shape": "openai"}]
    assert admission["unit_billing"] == [{"variant": "edit", "input_images": 3}]
    request_id = str(admission["request_id"])
    started = json.loads(
        control.start_attempt(json.dumps({"request_id": request_id, "attempt_ordinal": 0}))
    )
    billed = {"kind": "image", "variant": "edit", "quantity_milli": 2000}
    billed["input_quantity_milli"] = 3000
    settled = control.settle(
        json.dumps(
            {
                "request_id": request_id,
                "attempt_id": started["attempt_id"],
                "outcome": "completed",
                "usage": {"input_tokens": None, "output_tokens": None, "billed_units": billed},
                "tool_names": [],
                "failure": None,
            }
        )
    )
    assert settled == "{}"
    ledger = cast("SQLiteAttemptLedger", control._components.ledger)  # noqa: SLF001
    with sqlite3.connect(ledger.database_path) as connection:
        row = connection.execute(
            "select estimated_cost_nano_usd, billed_input_quantity_milli, billed_unit_rate "
            "from gateway_attempts where request_id = ?",
            (request_id,),
        ).fetchone()
    # 2 edited images at $0.06 plus 3 source images at $0.01.
    assert row == (150_000_000, 3000, 60_000_000)


@pytest.mark.parametrize(
    "units",
    [
        GatewayUnitPrices(kind=BilledUnitKind.IMAGE, rates={"": 40_000_000}),
        GatewayUnitPrices(kind=BilledUnitKind.IMAGE, rates={"": 40_000_000, "edit": 60_000_000}),
        GatewayUnitPrices(
            kind=BilledUnitKind.IMAGE, rates={"": 40_000_000}, input_rates={"": 10_000_000}
        ),
    ],
)
def test_an_xai_edit_rung_without_both_rates_is_not_admitted(
    tmp_path: Path, units: GatewayUnitPrices
) -> None:
    """Never at the generation rate, never with the source images unbilled."""
    control, raw_key = _edit_plane(
        tmp_path,
        provider="openai-compatible",
        provider_model="grok-imagine-image-2.0",
        base_url="https://api.x.ai/v1",
        prices=GatewayTokenPrices(units=units),
    )
    error = _edit_error(control, raw_key, {"prompt": "combine"}, {"image": _PNG})
    assert error["code"] == "unsupported_capability"
    assert error["param"] == "model"


def test_an_xai_edit_refuses_a_mask_by_name(tmp_path: Path) -> None:
    """xAI edits take no mask; the rung is excluded naming it."""
    control, raw_key = _edit_plane(
        tmp_path,
        provider="openai-compatible",
        provider_model="grok-imagine-image-2.0",
        base_url="https://api.x.ai/v1",
        prices=GatewayTokenPrices(units=_GROK_IMAGINE_UNITS),
    )
    error = _edit_error(control, raw_key, {"prompt": "x"}, {"image": _PNG, "mask": _PNG})
    assert error["param"] == "mask"

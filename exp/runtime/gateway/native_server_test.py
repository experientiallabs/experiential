"""Tests for the shared native gateway process host."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

from exp.runtime.gateway import native_server
from exp.runtime.gateway.native_server import (
    PRE_FIRST_BYTE_MAX_ENV,
    PRE_FIRST_BYTE_RESERVE_ENV,
    NativeGatewayServerError,
    pre_first_byte_max_seconds,
    pre_first_byte_reserve_seconds,
    serve_native_gateway,
)


@pytest.mark.parametrize("public_only", [False, True])
def test_host_passes_the_serve_configuration(
    monkeypatch: pytest.MonkeyPatch, public_only: bool
) -> None:
    """The host serializes the exact serve configuration for the extension."""
    captured: dict[str, object] = {}

    def serve(
        control_plane: object,
        config_json: str,
        shutdown: object,
        on_listening: object,
        guardrail_detectors: object,
        capture: object,
        execution_bindings: object,
    ) -> None:
        """Capture the extension boundary call."""
        captured["control_plane"] = control_plane
        captured["config"] = json.loads(config_json)
        captured["shutdown"] = shutdown
        captured["on_listening"] = on_listening
        captured["guardrail_detectors"] = guardrail_detectors
        captured["capture"] = capture
        captured["execution_bindings"] = execution_bindings

    native = SimpleNamespace(serve=serve)
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)
    monkeypatch.setenv(PRE_FIRST_BYTE_RESERVE_ENV, "20")
    monkeypatch.setenv(PRE_FIRST_BYTE_MAX_ENV, "480")
    control = SimpleNamespace(guardrail_detectors={}, request_timeout_seconds=37.0)
    scopes = cast("native_server.ExecutionBindings", object())

    serve_native_gateway(
        control,
        host="0.0.0.0",
        port=8080,
        max_active_requests=23,
        execution_bindings=scopes,
        graceful_timeout_seconds=45.0,
        connect_timeout_seconds=4.0,
        public_upstreams_only=public_only,
        time_to_first_byte_seconds=12.0,
        native_usage_enabled=False,
        maximum_retained_output_bytes=512 * 1024 * 1024,
    )

    assert captured["control_plane"] is control
    assert captured["execution_bindings"] is scopes
    assert captured["shutdown"] is None
    config = cast("dict[str, object]", captured["config"])
    assert config["host"] == "0.0.0.0"
    assert config["port"] == 8080
    assert config["max_active_requests"] == 23
    assert "max_active_per_organization" not in config
    assert config["request_timeout_seconds"] == 37.0
    assert config["graceful_timeout_seconds"] == 45.0
    assert config["connect_timeout_seconds"] == 4.0
    assert config["public_upstreams_only"] is public_only
    assert config["time_to_first_byte_seconds"] == 12.0
    assert config["native_usage_enabled"] is False
    assert config["maximum_retained_output_bytes"] == 512 * 1024 * 1024
    assert config["pre_first_byte_reserve_seconds"] == 20.0
    assert config["pre_first_byte_max_seconds"] == 480.0
    assert captured["guardrail_detectors"] == {}
    assert captured["capture"] is None


def test_host_forwards_an_embedder_owned_shutdown_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provided stop handle reaches the extension beside the configuration."""
    captured: dict[str, object] = {}

    def serve(
        control_plane: object,
        config_json: str,
        shutdown: object,
        on_listening: object,
        guardrail_detectors: object,
        capture: object,
        execution_bindings: object,
    ) -> None:
        """Capture the extension boundary call."""
        del control_plane, config_json, on_listening, guardrail_detectors, capture
        captured["shutdown"] = shutdown

    native = SimpleNamespace(serve=serve)
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)
    handle = object()

    serve_native_gateway(
        SimpleNamespace(guardrail_detectors={}, request_timeout_seconds=12.0),
        host="127.0.0.1",
        port=8080,
        shutdown=cast("native_server.ShutdownHandle", handle),
    )

    assert captured["shutdown"] is handle


def test_host_maps_native_runtime_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native bind or runtime failure surfaces as the typed host error."""
    native = SimpleNamespace(serve=mock.Mock(side_effect=RuntimeError("bind failed")))
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)

    with pytest.raises(NativeGatewayServerError, match="bind failed"):
        serve_native_gateway(
            SimpleNamespace(guardrail_detectors={}, request_timeout_seconds=10.0),
            host="127.0.0.1",
            port=8080,
        )


def test_host_treats_native_keyboard_interrupt_as_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native SIGINT drain returns normally."""
    native = SimpleNamespace(serve=mock.Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)

    serve_native_gateway(
        SimpleNamespace(guardrail_detectors={}, request_timeout_seconds=10.0),
        host="127.0.0.1",
        port=8080,
    )


def test_host_requires_the_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing extension raises the typed host error with the module name."""

    def missing(_name: str) -> object:
        """Simulate an absent compiled extension."""
        raise ModuleNotFoundError("exp_gateway_native")

    monkeypatch.setattr(native_server.importlib, "import_module", missing)
    with pytest.raises(NativeGatewayServerError, match="not installed"):
        serve_native_gateway(
            SimpleNamespace(guardrail_detectors={}, request_timeout_seconds=10.0),
            host="127.0.0.1",
            port=8080,
        )


def test_the_reserve_override_is_optional_and_validated() -> None:
    """Unset keeps the data plane's default; a negative value fails loudly."""
    assert pre_first_byte_reserve_seconds({}) is None
    assert pre_first_byte_reserve_seconds({PRE_FIRST_BYTE_RESERVE_ENV: "7.5"}) == 7.5
    with pytest.raises(ValueError):
        pre_first_byte_reserve_seconds({PRE_FIRST_BYTE_RESERVE_ENV: "-1"})


def test_the_budget_ceiling_override_is_optional_and_validated() -> None:
    """Unset keeps the data plane's 600 s default; a negative value fails loudly."""
    assert pre_first_byte_max_seconds({}) is None
    assert pre_first_byte_max_seconds({PRE_FIRST_BYTE_MAX_ENV: "900"}) == 900.0
    with pytest.raises(ValueError):
        pre_first_byte_max_seconds({PRE_FIRST_BYTE_MAX_ENV: "-5"})

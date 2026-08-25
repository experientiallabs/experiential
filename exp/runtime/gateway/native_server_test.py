"""Tests for the shared native gateway process host."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

from exp.runtime.gateway import native_server
from exp.runtime.gateway.native_server import (
    NativeGatewayServerError,
    serve_native_gateway,
)


def test_host_passes_the_serve_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host serializes the exact serve configuration for the extension."""
    captured: dict[str, object] = {}

    def serve(
        control_plane: object,
        config_json: str,
        shutdown: object,
        on_listening: object,
    ) -> None:
        """Capture the extension boundary call."""
        captured["control_plane"] = control_plane
        captured["config"] = json.loads(config_json)
        captured["shutdown"] = shutdown
        captured["on_listening"] = on_listening

    native = SimpleNamespace(serve=serve)
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)
    control = SimpleNamespace(request_timeout_seconds=37.0)

    serve_native_gateway(
        control,
        host="0.0.0.0",
        port=8080,
        max_active_requests=23,
        graceful_timeout_seconds=45.0,
        native_usage_enabled=False,
    )

    assert captured["control_plane"] is control
    assert captured["shutdown"] is None
    config = cast("dict[str, object]", captured["config"])
    assert config["host"] == "0.0.0.0"
    assert config["port"] == 8080
    assert config["max_active_requests"] == 23
    assert config["request_timeout_seconds"] == 37.0
    assert config["graceful_timeout_seconds"] == 45.0
    assert config["native_usage_enabled"] is False


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
    ) -> None:
        """Capture the extension boundary call."""
        del control_plane, config_json, on_listening
        captured["shutdown"] = shutdown

    native = SimpleNamespace(serve=serve)
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)
    handle = object()

    serve_native_gateway(
        SimpleNamespace(request_timeout_seconds=12.0),
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
            SimpleNamespace(request_timeout_seconds=10.0),
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
        SimpleNamespace(request_timeout_seconds=10.0),
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
            SimpleNamespace(request_timeout_seconds=10.0),
            host="127.0.0.1",
            port=8080,
        )

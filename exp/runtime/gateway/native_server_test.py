"""Tests for the shared native gateway process host."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

from exp.runtime.gateway import native_server
from exp.runtime.gateway.native_server import (
    NativeGatewayServerError,
    serve_native_gateway,
)


class _FallbackServer:
    """Small Uvicorn stand-in that exposes startup and observes shutdown."""

    last: _FallbackServer | None = None

    def __init__(self, _config: object) -> None:
        """Record the latest server and initialize lifecycle flags."""
        self.started = False
        self.should_exit = False
        _FallbackServer.last = self

    def run(self, *, sockets: list[object]) -> None:
        """Mark started and remain alive until the host requests shutdown."""
        assert len(sockets) == 1
        self.started = True
        while not self.should_exit:
            time.sleep(0.001)


def test_host_passes_public_and_fallback_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rust owns the public listener while Python receives an allocated loopback port."""
    captured: dict[str, object] = {}

    def serve(control_plane: object, config_json: str) -> None:
        """Capture the extension boundary call."""
        captured["control_plane"] = control_plane
        captured["config"] = json.loads(config_json)

    native = SimpleNamespace(serve=serve)
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)
    monkeypatch.setattr(native_server.uvicorn, "Server", _FallbackServer)
    control = SimpleNamespace(request_timeout_seconds=37.0)
    fallback_app = mock.Mock()

    serve_native_gateway(
        fallback_app,
        control,
        host="0.0.0.0",
        port=8080,
        max_active_requests=23,
        graceful_timeout_seconds=45.0,
        native_usage_enabled=False,
    )

    assert captured["control_plane"] is control
    config = cast("dict[str, object]", captured["config"])
    assert config["host"] == "0.0.0.0"
    assert config["port"] == 8080
    assert config["max_active_requests"] == 23
    assert config["request_timeout_seconds"] == 37.0
    assert config["graceful_timeout_seconds"] == 45.0
    assert config["native_usage_enabled"] is False
    assert isinstance(config["fallback_port"], int)
    assert _FallbackServer.last is not None and _FallbackServer.last.should_exit


def test_host_maps_native_runtime_failures_and_stops_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native bind or runtime failure closes the embedded Python server."""
    native = SimpleNamespace(serve=mock.Mock(side_effect=RuntimeError("bind failed")))
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)
    monkeypatch.setattr(native_server.uvicorn, "Server", _FallbackServer)

    with pytest.raises(NativeGatewayServerError, match="bind failed"):
        serve_native_gateway(
            mock.Mock(),
            SimpleNamespace(request_timeout_seconds=10.0),
            host="127.0.0.1",
            port=8080,
        )
    assert _FallbackServer.last is not None and _FallbackServer.last.should_exit


def test_host_treats_native_keyboard_interrupt_as_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native SIGINT drain returns normally and stops the fallback server."""
    native = SimpleNamespace(serve=mock.Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(native_server.importlib, "import_module", lambda _name: native)
    monkeypatch.setattr(native_server.uvicorn, "Server", _FallbackServer)

    serve_native_gateway(
        mock.Mock(),
        SimpleNamespace(request_timeout_seconds=10.0),
        host="127.0.0.1",
        port=8080,
    )

    assert _FallbackServer.last is not None and _FallbackServer.last.should_exit

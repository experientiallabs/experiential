"""Checks for explicit native local capture configuration and real delivery."""

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.local_capture import open_local_capture
from exp.runtime.gateway.local_capture_config import CaptureBinding, CaptureConfiguration
from exp.runtime.gateway.local_capture_contracts import CapturePolicy, LocalCaptureScope
from exp.runtime.gateway.local_capture_store import LocalCaptureStore
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_server import serve_native_gateway
from exp.runtime.gateway.tests.launch_test import (
    _configure_gateway,
    _LoopbackProvider,
    _unused_port,
    _wait_ready,
)

exp_gateway_native = pytest.importorskip("exp_gateway_native")


def test_capture_bindings_cannot_ambiguously_assign_an_application(tmp_path: Path) -> None:
    """A single request cannot acquire two different application authorities."""
    binding = CaptureBinding(
        alias="model",
        policy=CapturePolicy(scope=LocalCaptureScope(user_id="user", application_id="app")),
    )
    with pytest.raises(ValidationError):
        CaptureConfiguration(database_path=tmp_path / "experience.db", bindings=(binding, binding))


@pytest.mark.parametrize("enabled,ghost", [(True, False), (False, False), (True, True)])
def test_real_native_gateway_capture_is_opt_in_and_ghost_stays_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, ghost: bool
) -> None:
    """Official protocol bodies cross real sockets and survive native-to-Python storage."""
    monkeypatch.setenv("LOOPBACK_PROVIDER_KEY", "provider-secret")
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    _manager, raw_key = _configure_gateway(
        tmp_path, base_url=f"http://127.0.0.1:{provider.server_port}/v1"
    )
    components = load_gateway_components(tmp_path)
    _organization, user_id = components.store.authenticated_identity(raw_key=raw_key)
    scope = LocalCaptureScope(user_id=user_id, application_id="claims-agent")
    database = tmp_path / "experiences.sqlite3"
    capture = CaptureConfiguration(
        database_path=database,
        bindings=(
            CaptureBinding(alias="coding", policy=CapturePolicy(scope=scope, enabled=enabled)),
        ),
    )
    controller = open_local_capture(None if ghost else capture)
    port = _unused_port()
    shutdown = exp_gateway_native.shutdown_handle()
    failures: list[BaseException] = []

    def run() -> None:
        """Run the native host with explicit scoped capture, retaining startup errors."""
        try:
            serve_native_gateway(
                NativeControlPlane(components, capture=controller),
                host="127.0.0.1",
                port=port,
                capture=None if controller is None else controller.native,
                shutdown=shutdown,
            )
        except BaseException as error:  # noqa: BLE001 - assert the error after thread shutdown.
            failures.append(error)

    gateway_thread = threading.Thread(target=run, daemon=True)
    gateway_thread.start()
    try:
        _wait_ready(port, gateway_thread)
        for surface in ("chat/completions", "responses"):
            for stream in (False, True):
                payload = {"model": "coding", "stream": stream}
                if surface == "responses":
                    payload["input"] = "hello capture"
                else:
                    payload["messages"] = [{"role": "user", "content": "hello capture"}]
                response = httpx.post(
                    f"http://127.0.0.1:{port}/v1/{surface}",
                    json=payload,
                    headers={"Authorization": f"Bearer {raw_key}"},
                    timeout=10,
                )
                assert response.status_code == 200, response.text
                assert "hello" in response.text
    finally:
        shutdown.request_shutdown()
        gateway_thread.join(timeout=10)
        components.write_ledger.close()
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
    assert not failures
    assert not gateway_thread.is_alive()
    if enabled and not ghost:
        experiences = LocalCaptureStore(database, scope).read_after()
        assert len(experiences) == 4
        assert {row.experience.protocol for row in experiences} == {"chat_completions", "responses"}
        assert all("exact_tokens" not in row.experience.model_dump() for row in experiences)
        assert all(row.experience.scope == scope for row in experiences)
        serialized = "".join(row.experience.model_dump_json() for row in experiences)
        assert raw_key not in serialized
        assert "provider-secret" not in serialized
        assert (
            LocalCaptureStore(
                database, LocalCaptureScope(user_id=user_id, application_id="other")
            ).read_after()
            == ()
        )
    else:
        assert not database.exists()


def test_aliases_sharing_application_cannot_conflict_on_retention(tmp_path: Path) -> None:
    """A second alias cannot silently shorten another alias's retained evidence."""
    scope = LocalCaptureScope(user_id="user", application_id="app")
    policy = CapturePolicy(scope=scope, enabled=True)
    with pytest.raises(ValueError, match="share one capture policy"):
        CaptureConfiguration(
            database_path=tmp_path / "capture.sqlite",
            bindings=(
                CaptureBinding(alias="first", policy=policy),
                CaptureBinding(
                    alias="second", policy=policy.model_copy(update={"retention_seconds": 1})
                ),
            ),
        )

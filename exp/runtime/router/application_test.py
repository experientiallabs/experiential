"""Tests for project selection and the gateway-backed Python compatibility client."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from openai import OpenAI

from exp.common.routing import RoutingDecision
from exp.runtime.gateway.native_server import NativeGatewayServerError
from exp.runtime.gateway.project_alias import ProjectGatewayAlias
from exp.runtime.router.application import RouterApplicationError, load_router


class _FakeShutdownHandle:
    """Stand-in for the native stop handle observable from a fake server."""

    def __init__(self) -> None:
        """Create the unsignalled stop event."""
        self.requested = threading.Event()

    def request_shutdown(self) -> None:
        """Signal the fake server loop to stop."""
        self.requested.set()


class _GatewayStub(BaseHTTPRequestHandler):
    """Answer the two routes the compatibility client exercises."""

    def do_GET(self) -> None:  # noqa: N802 - http.server contract.
        """Serve liveness and the smallest official model list."""
        if self.path == "/health/live":
            body = json.dumps({"status": "live"}).encode()
        elif self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": []}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - contract.
        """Silence the default stderr access log."""


class _Management:
    """Capture the virtual-key lifecycle owned by the compatibility client."""

    issued: list[tuple[str, str]] = []
    revoked: list[str] = []
    expected_root: Path | None = None

    def __init__(self, root: Path) -> None:
        """Require the requested EXP root."""
        assert root == type(self).expected_root
        self.root = root

    def issue_key(self, *, identity_id: str, key_id: str) -> object:
        """Return one synthetic raw key and record its owner."""
        type(self).issued.append((identity_id, key_id))
        return SimpleNamespace(raw_key="exp_test_key")

    def revoke_key(self, *, key_id: str) -> bool:
        """Record exact key revocation."""
        type(self).revoked.append(key_id)
        return True


def _install_common_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    prepared: list[tuple[str, Path, str | None]],
    loaded: list[tuple[Path, object, frozenset[str] | None]],
    components: object,
) -> None:
    """Stub the alias preparation, key lifecycle, and component loading seams.

    Args:
        monkeypatch: Active patch context.
        tmp_path: Expected EXP root.
        prepared: Sink recording alias-preparation calls.
        loaded: Sink recording component-loading calls.
        components: Component stub the loader returns.
    """
    _Management.issued = []
    _Management.revoked = []
    _Management.expected_root = tmp_path

    def prepare(
        project: str,
        root: Path,
        *,
        policy_id: str | None,
        project_repository: object,
        environment: object,
        runtime_catalog: object,
    ) -> ProjectGatewayAlias:
        """Return one already activated project-backed alias."""
        del environment, project_repository, runtime_catalog
        prepared.append((project, root, policy_id))
        return ProjectGatewayAlias(
            alias=project,
            alias_revision_id="revision-a",
            identity_id="project-identity",
            policy_id="policy-a",
            changed=True,
        )

    def load_components(
        root: Path,
        *,
        environment: object,
        project_repository: object,
        decision_sink: object,
        only_aliases: frozenset[str] | None,
    ) -> object:
        """Return the component stub used by the owned native plane."""
        del environment, project_repository
        loaded.append((root, decision_sink, only_aliases))
        return components

    monkeypatch.setattr("exp.runtime.router.application.GatewayManagement", _Management)
    monkeypatch.setattr(
        "exp.runtime.router.application.prepare_project_gateway_alias",
        prepare,
    )
    monkeypatch.setattr(
        "exp.runtime.router.application.load_gateway_components",
        load_components,
    )
    monkeypatch.setattr(
        "exp.runtime.router.application.native_serving_blockers",
        lambda _components: (),
    )
    monkeypatch.setattr(
        "exp.runtime.router.application.load_guardrail_engine",
        lambda _root: None,
    )
    monkeypatch.setattr(
        "exp.runtime.router.application.NativeControlPlane",
        lambda components, **_kwargs: SimpleNamespace(components=components),
    )
    monkeypatch.setattr(
        "exp.runtime.router.application.importlib",
        SimpleNamespace(
            import_module=lambda _name: SimpleNamespace(
                shutdown_handle=_FakeShutdownHandle,
                metrics_snapshot_json=lambda: "{}",
            )
        ),
    )


def test_load_router_serves_the_native_gateway_and_revokes_its_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility client owns one loopback native plane end to end."""
    prepared: list[tuple[str, Path, str | None]] = []
    loaded: list[tuple[Path, object, frozenset[str] | None]] = []
    components = SimpleNamespace(
        selection_workers=SimpleNamespace(shutdown=lambda: shut.append("workers")),
        write_ledger=SimpleNamespace(close=lambda: shut.append("ledger")),
    )
    shut: list[str] = []
    _install_common_stubs(
        monkeypatch, tmp_path, prepared=prepared, loaded=loaded, components=components
    )

    def fake_serve(
        control_plane: object,
        *,
        host: str,
        port: int,
        shutdown: _FakeShutdownHandle | None = None,
        **_kwargs: object,
    ) -> None:
        """Serve a real loopback HTTP stub until the stop handle fires."""
        assert cast("SimpleNamespace", control_plane).components is components
        assert shutdown is not None
        server = ThreadingHTTPServer((host, port), _GatewayStub)
        pump = threading.Thread(target=server.serve_forever, daemon=True)
        pump.start()
        try:
            assert shutdown.requested.wait(timeout=30)
        finally:
            server.shutdown()

    monkeypatch.setattr("exp.runtime.router.application.serve_native_gateway", fake_serve)

    def decision_sink(_decision: RoutingDecision) -> None:
        """Accept one served project selection."""

    client = load_router(
        "support",
        root=tmp_path,
        policy_id="policy-a",
        decision_sink=decision_sink,
    )
    assert isinstance(client, OpenAI)
    assert client.models.list().data == []
    client.close()

    assert prepared == [("support", tmp_path, "policy-a")]
    assert loaded == [(tmp_path, decision_sink, frozenset({"support"}))]
    assert _Management.issued[0][0] == "project-identity"
    assert _Management.revoked == [_Management.issued[0][1]]
    assert shut == ["workers", "ledger"]


def test_load_router_startup_failure_stops_and_revokes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serving failure surfaces as the typed error and revokes the key."""
    components = SimpleNamespace(
        selection_workers=SimpleNamespace(shutdown=lambda: None),
        write_ledger=SimpleNamespace(close=lambda: None),
    )
    _install_common_stubs(monkeypatch, tmp_path, prepared=[], loaded=[], components=components)

    def failing_serve(*_args: object, **_kwargs: object) -> None:
        """Fail the native bind immediately."""
        raise NativeGatewayServerError("the native gateway failed: bind refused")

    monkeypatch.setattr("exp.runtime.router.application.serve_native_gateway", failing_serve)

    with pytest.raises(RouterApplicationError, match="failed to start"):
        load_router("support", root=tmp_path)
    assert len(_Management.issued) == 1
    assert _Management.revoked == [_Management.issued[0][1]]


def test_ghost_compatibility_rejects_a_persistent_project_decision_sink(tmp_path: Path) -> None:
    """Ghost compatibility cannot silently persist project-selection decisions."""

    def decision_sink(_decision: object) -> None:
        """Accept one decision for the rejected option combination."""

    with pytest.raises(RouterApplicationError, match="ghost mode cannot use"):
        load_router("support", root=tmp_path, ghost=True, decision_sink=decision_sink)

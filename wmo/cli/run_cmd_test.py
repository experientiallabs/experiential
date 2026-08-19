"""Development-only loopback run-command tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.cli.gateway.compatibility import ProjectGatewayCompatibility


@pytest.mark.parametrize("ghost", [False, True])
def test_project_form_launches_the_normal_gateway_on_loopback(
    monkeypatch: pytest.MonkeyPatch,
    ghost: bool,
) -> None:
    """Both project compatibility modes launch the ordinary gateway application.

    Args:
        monkeypatch: Scoped replacements for runtime and server boundaries.
        ghost: Whether the invocation disables durable interaction state.
    """
    prepared: list[tuple[str, Path, str | None]] = []
    loaded: list[tuple[Path, frozenset[str] | None]] = []
    served: list[tuple[object, str, int]] = []
    application = object()

    def prepare(project: str, root: Path, *, policy_id: str | None) -> ProjectGatewayCompatibility:
        """Return one already materialized project-backed gateway alias.

        Args:
            project: Requested project identifier.
            root: Requested gateway and artifact root.
            policy_id: Optional exact policy selection.

        Returns:
            Compatibility authority consumed by the shared launch path.
        """
        prepared.append((project, root, policy_id))
        return ProjectGatewayCompatibility(
            alias=project,
            alias_revision_id="revision-a",
            identity_id="project-identity",
            key_file=root / "gateway" / "compatibility-keys" / "project-identity.txt",
            policy_id="policy-a",
            changed=True,
        )

    async def preflight() -> None:
        """Complete the gateway preflight without provider work."""

    def load_gateway(
        root: Path,
        *,
        graceful_timeout_seconds: float,
        project_loader: object,
        only_aliases: frozenset[str] | None,
    ) -> object:
        """Capture the one shared lifecycle composition.

        Args:
            root: Gateway and artifact root.
            graceful_timeout_seconds: Requested shutdown drain bound.
            project_loader: Injected selection-only project loader.
            only_aliases: Optional compatibility alias filter.

        Returns:
            Gateway runtime fixture passed to the normal server.
        """
        del graceful_timeout_seconds, project_loader
        loaded.append((root, only_aliases))
        return SimpleNamespace(
            app=application,
            service=SimpleNamespace(preflight=preflight),
            reconciled_expired_requests=0,
            reconciled_unknown_attempts=0,
        )

    @contextmanager
    def instance_lock(root: Path, *, port: int) -> Iterator[None]:
        """Capture and provide the local single-instance boundary.

        Args:
            root: Gateway root.
            port: Requested loopback port.

        Yields:
            Control while the test owns the synthetic lock.
        """
        assert root == Path("/tmp/local-wmo")
        assert port == 8123
        yield

    def serve(value: object, *, host: str, port: int) -> None:
        """Capture the application and loopback bind without starting a server.

        Args:
            value: Composed ASGI application.
            host: Required loopback host.
            port: Requested local port.
        """
        served.append((value, host, port))

    monkeypatch.setattr("wmo.cli.gateway.compatibility.prepare_project_gateway", prepare)
    monkeypatch.setattr("wmo.runtime.gateway.lifecycle.load_local_gateway", load_gateway)
    monkeypatch.setattr("wmo.runtime.gateway.lifecycle.gateway_instance_lock", instance_lock)
    monkeypatch.setattr("uvicorn.run", serve)

    arguments = [
        "run",
        "project-a",
        "--root",
        "/tmp/local-wmo",
        "--policy",
        "policy-a",
        "--port",
        "8123",
        "--json",
    ]
    if ghost:
        arguments.append("--ghost")
    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert prepared == [("project-a", Path("/tmp/local-wmo"), "policy-a")]
    assert loaded == [(Path("/tmp/local-wmo"), frozenset({"project-a"}))]
    assert served == [(application, "127.0.0.1", 8123)]
    receipt = json.loads(result.stdout)
    assert receipt["launch_mode"] == "project_alias"
    assert receipt["base_url"] == "http://127.0.0.1:8123/v1"
    assert receipt["project_alias"] == "project-a"
    assert receipt["gateway_accounting"] == "enabled"
    assert "--host" not in CliRunner().invoke(app, ["run", "--help"]).output


def test_no_argument_noninteractive_run_returns_stable_empty_state_json(tmp_path: Path) -> None:
    """Automation receives exact setup commands instead of prompts or runtime seeds."""
    result = CliRunner().invoke(
        app,
        ["run", "--root", str(tmp_path), "--non-interactive", "--json"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "gateway_not_initialized"
    assert payload["error"]["next_commands"][0].startswith("wmo config gateway init")
    assert not (tmp_path / "gateway").exists()

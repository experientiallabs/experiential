"""Development-only loopback run-command tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.project import ProjectStore


@pytest.mark.parametrize("ghost", [False, True])
def test_run_loads_once_and_can_only_bind_loopback(
    monkeypatch: pytest.MonkeyPatch,
    ghost: bool,
) -> None:
    """The CLI activates locally, supports ghost mode, and only binds loopback.

    Args:
        monkeypatch: Scoped replacements for runtime and server boundaries.
        ghost: Whether the invocation disables durable interaction state.
    """
    runtime = SimpleNamespace(
        policy=SimpleNamespace(policy_id="policy-a", judgment_status="provisional")
    )
    loaded: list[tuple[str, object, str | None]] = []
    served: list[tuple[object, str, int]] = []
    application = object()

    def load(project: str, root: object, *, policy_id: str | None = None) -> object:
        """Capture the requested project activation without model construction.

        Args:
            project: Requested project identifier.
            root: Requested local artifact root.
            policy_id: Optional exact policy selection.

        Returns:
            Provider-idle runtime fixture.
        """
        loaded.append((project, root, policy_id))
        return runtime

    services: list[object] = []
    ghost_modes: list[bool] = []

    def compose(store: ProjectStore, selected: object, *, ghost: bool = False) -> object:
        """Capture project-scoped traffic composition without dispatching the runtime.

        Args:
            store: Project store created for the requested root and project.
            selected: Previously loaded frozen runtime.
            ghost: Whether durable interaction state is disabled.

        Returns:
            Opaque completion service passed to application composition.
        """
        assert store.paths.project_id == "project-a"
        assert store.paths.root == Path("/tmp/local-wmo")
        assert selected is runtime
        ghost_modes.append(ghost)
        service = object()
        services.append(service)
        return service

    def create(
        project: str,
        selected: object,
        *,
        completion_service: object,
    ) -> object:
        """Capture low-level application injection selected by the run command.

        Args:
            project: Public endpoint model name.
            selected: Loaded frozen runtime.
            completion_service: Project-scoped durable completion owner.

        Returns:
            Opaque application passed to the server.
        """
        assert (project, selected) == ("project-a", runtime)
        assert completion_service is services[0]
        return application

    def serve(value: object, *, host: str, port: int) -> None:
        """Capture the application and loopback bind without starting a server.

        Args:
            value: Composed ASGI application.
            host: Required loopback host.
            port: Requested local port.
        """
        served.append((value, host, port))

    monkeypatch.setattr("wmo.runtime.router.application.load_project_router", load)
    monkeypatch.setattr("wmo.runtime.router.application.create_project_completion_service", compose)
    monkeypatch.setattr("wmo.runtime.router.application.create_project_router_app", create)
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
    ]
    if ghost:
        arguments.append("--ghost")
    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert loaded == [("project-a", Path("/tmp/local-wmo"), "policy-a")]
    assert len(services) == 1
    assert ghost_modes == [ghost]
    assert served == [(application, "127.0.0.1", 8123)]
    assert "provisional judgment" in result.output
    assert "OpenAI API router at http://127.0.0.1:8123/v1" in result.output
    assert ("ghost mode enabled" in result.output) is ghost
    assert "--host" not in CliRunner().invoke(app, ["run", "--help"]).output

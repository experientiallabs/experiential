"""Development-only loopback run-command tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app


def test_run_loads_once_and_can_only_bind_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI activates locally without a request and never accepts a remote host."""
    runtime = SimpleNamespace(
        policy=SimpleNamespace(policy_id="policy-a", judgment_status="provisional")
    )
    loaded: list[tuple[str, object, str | None]] = []
    served: list[tuple[object, str, int]] = []
    application = object()

    def load(project: str, root: object, *, policy_id: str | None = None) -> object:
        loaded.append((project, root, policy_id))
        return runtime

    def create(project: str, selected: object) -> object:
        assert (project, selected) == ("project-a", runtime)
        return application

    def serve(value: object, *, host: str, port: int) -> None:
        served.append((value, host, port))

    monkeypatch.setattr("wmo.runtime.router.application.load_project_router", load)
    monkeypatch.setattr("wmo.runtime.router.application.create_project_router_app", create)
    monkeypatch.setattr("uvicorn.run", serve)

    result = CliRunner().invoke(
        app,
        ["run", "project-a", "--root", "/tmp/local-wmo", "--policy", "policy-a", "--port", "8123"],
    )

    assert result.exit_code == 0, result.output
    assert loaded == [("project-a", Path("/tmp/local-wmo"), "policy-a")]
    assert served == [(application, "127.0.0.1", 8123)]
    assert "provisional judgment" in result.output
    assert "OpenAI API router at http://127.0.0.1:8123/v1" in result.output
    assert "--host" not in CliRunner().invoke(app, ["run", "--help"]).output

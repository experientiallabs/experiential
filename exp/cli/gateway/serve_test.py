"""Loopback gateway serving tests for the root default flow."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

import exp.cli.gateway.serve as run_app
from exp.cli.app import app
from exp.cli.gateway.compatibility import ProjectGatewayCompatibility
from exp.cli.gateway.setup import InteractiveSetupResult
from exp.runtime.gateway.auth import IssuedVirtualKey
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError


def test_run_command_starts_the_gateway_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The explicit run command forwards its arguments to the shared gateway launcher."""
    captured: list[
        tuple[
            str | None,
            Path,
            str | None,
            int,
            bool,
            bool,
            bool,
            bool,
            float,
            int,
        ]
    ] = []

    def capture_start(
        *,
        project: str | None,
        root: Path,
        policy: str | None,
        port: int,
        ghost: bool,
        non_interactive: bool,
        json_output: bool,
        check: bool,
        graceful_timeout: float,
        max_active_requests: int,
    ) -> None:
        """Capture the shared launcher arguments without starting a server."""
        captured.append(
            (
                project,
                root,
                policy,
                port,
                ghost,
                non_interactive,
                json_output,
                check,
                graceful_timeout,
                max_active_requests,
            )
        )

    monkeypatch.setattr(run_app, "start_gateway", capture_start)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "project-a",
            "--root",
            str(tmp_path),
            "--policy",
            "policy-a",
            "--port",
            "8123",
            "--ghost",
            "--non-interactive",
            "--json",
            "--check",
            "--graceful-timeout",
            "4.5",
            "--max-active-requests",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [
        (
            "project-a",
            tmp_path,
            "policy-a",
            8123,
            True,
            True,
            True,
            True,
            4.5,
            7,
        )
    ]


@pytest.mark.parametrize("ghost", [False, True])
def test_project_option_launches_the_native_gateway_on_loopback(
    monkeypatch: pytest.MonkeyPatch,
    ghost: bool,
) -> None:
    """Both project compatibility modes launch the ordinary native gateway.

    Args:
        monkeypatch: Scoped replacements for runtime and server boundaries.
        ghost: Whether the invocation disables durable interaction state.
    """
    prepared: list[tuple[str, Path, str | None]] = []
    loaded: list[tuple[Path, frozenset[str] | None]] = []
    served: list[tuple[object, str, int]] = []
    control_planes: list[object] = []

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

    components = SimpleNamespace(
        reconciled_expired_requests=0,
        reconciled_unknown_attempts=0,
        unavailable_aliases=(),
    )

    def load_components(
        root: Path,
        *,
        project_repository: object,
        only_aliases: frozenset[str] | None,
    ) -> object:
        """Capture the one shared engine-neutral component load.

        Args:
            root: Gateway and artifact root.
            project_repository: Injected immutable project activation repository.
            only_aliases: Optional compatibility alias filter.

        Returns:
            Component fixture consumed by the native control plane.
        """
        del project_repository
        loaded.append((root, only_aliases))
        return components

    def control_plane_factory(value: object, **_kwargs: object) -> object:
        """Capture the composed control plane over the loaded components."""
        assert value is components
        plane = SimpleNamespace(
            components=value,
            reconciled_expired_requests=0,
            reconciled_unknown_attempts=0,
        )
        control_planes.append(plane)
        return plane

    @contextmanager
    def instance_lock(root: Path, *, port: int) -> Iterator[None]:
        """Capture and provide the local single-instance boundary.

        Args:
            root: Gateway root.
            port: Requested loopback port.

        Yields:
            Control while the test owns the synthetic lock.
        """
        assert root == Path("/tmp/local-exp")
        assert port == 8123
        yield

    def serve(value: object, *, host: str, port: int, **_kwargs: object) -> None:
        """Capture the control plane and loopback bind without serving.

        Args:
            value: Composed native control plane.
            host: Required loopback host.
            port: Requested local port.
        """
        served.append((value, host, port))

    monkeypatch.setattr("exp.cli.gateway.compatibility.prepare_project_gateway", prepare)
    monkeypatch.setattr("exp.runtime.gateway.lifecycle.load_gateway_components", load_components)
    monkeypatch.setattr("exp.runtime.gateway.lifecycle.gateway_instance_lock", instance_lock)
    monkeypatch.setattr(
        "exp.runtime.gateway.native_execution.native_serving_blockers",
        lambda _components: (),
    )
    monkeypatch.setattr(
        "exp.runtime.gateway.guardrails.config.load_guardrail_engine",
        lambda _root: None,
    )
    monkeypatch.setattr(
        "exp.runtime.gateway.native_bridge.NativeControlPlane",
        control_plane_factory,
    )
    monkeypatch.setattr("exp.runtime.gateway.native_server.serve_native_gateway", serve)

    arguments = [
        "--project",
        "project-a",
        "--root",
        "/tmp/local-exp",
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
    assert prepared == [("project-a", Path("/tmp/local-exp"), "policy-a")]
    assert loaded == [(Path("/tmp/local-exp"), frozenset({"project-a"}))]
    assert served == [(control_planes[0], "127.0.0.1", 8123)]
    receipt = json.loads(result.stdout)
    assert receipt["launch_mode"] == "project_alias"
    assert receipt["base_url"] == "http://127.0.0.1:8123/v1"
    assert receipt["project_alias"] == "project-a"
    assert receipt["gateway_accounting"] == "enabled"
    assert "--host" not in CliRunner().invoke(app, ["--help"]).output
    assert "--engine" not in CliRunner().invoke(app, ["--help"]).output


def test_noninteractive_default_gateway_returns_stable_empty_state_json(tmp_path: Path) -> None:
    """Automation receives exact setup commands instead of prompts or runtime seeds."""
    result = CliRunner().invoke(
        app,
        ["--root", str(tmp_path), "--non-interactive", "--json"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "gateway_not_initialized"
    assert payload["error"]["next_commands"][0].startswith("exp config gateway init")
    assert not (tmp_path / "gateway").exists()


def test_unavailable_alias_entries_shape_the_startup_receipt() -> None:
    """Failed aliases become JSON-ready receipt entries naming alias and reason."""
    entries = run_app._unavailable_alias_entries((("broken", "missing MISSING_PROVIDER_KEY"),))
    assert entries == [{"alias": "broken", "reason": "missing MISSING_PROVIDER_KEY"}]
    assert run_app._unavailable_alias_entries(()) == []


def test_emit_unavailable_aliases_warns_with_alias_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Human startup output names each failed alias, its 503 behavior, and reason."""
    buffer = io.StringIO()
    monkeypatch.setattr(run_app, "_console", Console(file=buffer, width=200))

    run_app._emit_unavailable_aliases((("broken", "missing MISSING_PROVIDER_KEY"),))

    output = buffer.getvalue()
    assert "'broken'" in output
    assert "503 unavailable_route" in output
    assert "missing MISSING_PROVIDER_KEY" in output


def test_missing_extension_is_an_actionable_error(tmp_path: Path) -> None:
    """A launch without the built extension names the exact build step."""
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def missing_extension(name: str, package: str | None = None) -> object | None:
        if name == "exp_gateway_native":
            return None
        return real_find_spec(name, package)

    with mock.patch.object(importlib.util, "find_spec", side_effect=missing_extension):
        result = CliRunner().invoke(app, ["--root", str(tmp_path)])
    assert result.exit_code == 2
    assert "exp_gateway_native" in result.output


def test_engine_flag_is_gone(tmp_path: Path) -> None:
    """The retired --engine flag is a usage error, not a silent no-op."""
    result = CliRunner().invoke(app, ["--root", str(tmp_path), "--engine", "rust"])
    assert result.exit_code == 2
    # Rich may interleave style escapes inside the echoed option name, so the
    # assertion runs over the escape-stripped output.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--engine" in plain


def test_first_run_delivers_credentials_before_readiness_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run credentials remain recoverable when local readiness rejects the route."""
    raw_key = "exp_vk_test-first-run-key"
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, no_color=True, highlight=False)
    setup_result = InteractiveSetupResult(
        identity_id="default",
        alias="gpt-5-6-luna",
        raw_key=raw_key,
        guardrails="Off",
    )

    def interactive_setup(root: Path) -> InteractiveSetupResult:
        """Return the setup result that the launch path must deliver before preflight."""
        assert root == tmp_path
        return setup_result

    def load_components(
        root: Path,
        *,
        project_repository: object,
        only_aliases: frozenset[str] | None,
    ) -> object:
        """Fail readiness only after confirming that setup credentials were printed."""
        del root, project_repository, only_aliases
        transcript = output.getvalue()
        assert transcript.index(f"export EXP_GATEWAY_KEY={raw_key}") < len(transcript)
        raise ValueError(
            "no granted active alias is locally available: "
            "gpt-5-6-luna (connection credential environment variable "
            "'OPENAI_API_KEY' is not set); run "
            "'OPENAI_API_KEY=YOUR_API_KEY exp'"
        )

    @contextmanager
    def instance_lock(root: Path, *, port: int) -> Iterator[None]:
        """Provide the single-process seam without creating a real gateway lock."""
        del root, port
        yield

    monkeypatch.setattr(run_app, "_console", console)
    monkeypatch.setattr("exp.cli.gateway.setup.interactive_gateway_setup", interactive_setup)
    monkeypatch.setattr("exp.runtime.gateway.lifecycle.load_gateway_components", load_components)
    monkeypatch.setattr("exp.runtime.gateway.lifecycle.gateway_instance_lock", instance_lock)
    monkeypatch.setattr(run_app.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(run_app.sys, "stdout", SimpleNamespace(isatty=lambda: True))

    with pytest.raises(typer.BadParameter, match="OPENAI_API_KEY") as failure:
        run_app._run_gateway(
            project=None,
            root=tmp_path,
            policy=None,
            port=8000,
            ghost=False,
            non_interactive=False,
            json_output=False,
            check=True,
            graceful_timeout=10.0,
        )

    transcript = output.getvalue()
    assert transcript.startswith(
        "███████╗██╗  ██╗██████╗ \n"
        "██╔════╝╚██╗██╔╝██╔══██╗\n"
        "█████╗   ╚███╔╝ ██████╔╝\n"
        "██╔══╝   ██╔██╗ ██╔═══╝ \n"
        "███████╗██╔╝ ██╗██║     \n"
        "╚══════╝╚═╝  ╚═╝╚═╝     \n"
    )
    assert "export EXP_GATEWAY_URL=http://127.0.0.1:8000/v1" in transcript
    assert f"export EXP_GATEWAY_KEY={raw_key}" in transcript
    assert "export OPENAI_API_KEY" not in transcript
    assert "First-run gateway setup completed, but the gateway is not ready." in transcript
    assert "run 'OPENAI_API_KEY=YOUR_API_KEY exp'" in str(failure.value)
    assert "exp config gateway key issue default --key-id RECOVERY_KEY --json" in transcript


def test_first_run_unknown_setup_outcome_delivers_the_only_raw_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving boundary preserves a key attached to an indeterminate setup commit."""
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, no_color=True, highlight=False)
    issued = IssuedVirtualKey(
        key_id="key-unknown",
        organization_id="local",
        identity_id="default",
        prefix="exp_vk_test",
        raw_key="exp_vk_unknown_secret",
        expires_at=None,
        created_at=datetime.now(UTC),
    )

    def interactive_setup(root: Path) -> InteractiveSetupResult:
        """Raise the typed uncertainty that must retain its one-time secret."""
        assert root == tmp_path
        raise AliasActivationOutcomeUnknownError(
            alias_id="default-gateway",
            revision_id="revision-unknown",
            issued=issued,
        )

    monkeypatch.setattr(run_app, "_console", console)
    monkeypatch.setattr("exp.cli.gateway.setup.interactive_gateway_setup", interactive_setup)
    monkeypatch.setattr(run_app.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(run_app.sys, "stdout", SimpleNamespace(isatty=lambda: True))

    with pytest.raises(typer.BadParameter, match="operation_outcome_unknown"):
        run_app._run_gateway(
            project=None,
            root=tmp_path,
            policy=None,
            port=8000,
            ghost=False,
            non_interactive=False,
            json_output=False,
            check=True,
            graceful_timeout=10.0,
        )

    transcript = output.getvalue()
    assert "export EXP_GATEWAY_KEY=exp_vk_unknown_secret" in transcript
    assert "Preserve this one-time gateway key: exp_vk_unknown_secret" in transcript
    assert "exp config gateway key issue default --key-id RECOVERY_KEY --json" in transcript

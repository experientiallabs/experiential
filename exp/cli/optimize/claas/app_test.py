"""Provider-free CLI preflight and explicit local/Modal backend selection."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer import Context
from typer.main import get_group
from typer.testing import CliRunner

from exp.cli.optimize.claas import app as commands
from exp.common.config.settings import set_maximum_command_cost_usd
from exp.optimize.claas.backends.modal.configuration import ModalLaunch
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.service.configuration import (
    BufferStatus,
    RunConfiguration,
    RunReport,
    RunStatus,
)
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration
from exp.optimize.claas.training_contracts_test import example, spec


def configuration(root: Path, *, mode: str = "burst", cost: float = 1) -> Path:
    """Write a provider-free launch input with finite fixture capacity."""
    launch = RunLaunchConfiguration(
        directory=root / "state",
        spec=spec(),
        run=RunConfiguration.model_validate({"mode": mode}),
        runtime=ResidentVerlSettings(
            checkpoint_root=root / "state" / "checkpoints", maximum_output_tokens=4
        ),
        compute_reservation_usd=cost,
    )
    path = root / "launch.json"
    path.write_text(launch.model_dump_json())
    return path


def report() -> RunReport:
    """Describe inert successful completion without pretending to run a GPU update."""
    return RunReport(
        status=RunStatus(
            mode="burst",
            state="closed",
            updates=0,
            policy_revision="policy-0",
            buffer=BufferStatus(),
        )
    )


def test_nested_commands_are_exact() -> None:
    """The learning surface exposes only burst and serve."""
    group = get_group(commands.claas_app)
    assert set(group.list_commands(Context(group))) == {"burst", "serve"}


@pytest.mark.parametrize("command,mode", [("burst", "run"), ("serve", "burst")])
def test_mode_mismatch_precedes_backend_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    mode: str,
) -> None:
    """A CLI spelling cannot silently rewrite the immutable run mode."""
    path = configuration(tmp_path, mode=mode)

    def forbidden(_name: str) -> None:
        """Fail if invalid preflight reaches any backend plugin."""
        pytest.fail("backend imported before mode validation")

    monkeypatch.setattr(commands.importlib, "import_module", forbidden)
    result = CliRunner().invoke(commands.claas_app, [command, str(path)])
    assert result.exit_code == 2
    assert "requires configuration.run.mode" in result.output


def test_budget_rejection_precedes_backend_import_even_with_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator estimate above the configured ceiling cannot import or launch a runtime."""
    path = configuration(tmp_path, cost=20)
    set_maximum_command_cost_usd(1, tmp_path / "budget")

    def forbidden(_name: str) -> None:
        """Fail if over-budget preflight reaches an optional compute backend."""
        pytest.fail("backend imported before budget authorization")

    monkeypatch.setattr(commands.importlib, "import_module", forbidden)
    result = CliRunner().invoke(
        commands.claas_app,
        [
            "burst",
            str(path),
            "--yes",
            "--root",
            str(tmp_path / "budget"),
        ],
    )
    assert result.exit_code == 2
    assert "exceeds the configured" in result.output


@pytest.mark.parametrize("command,mode", [("burst", "burst"), ("serve", "run")])
def test_local_invocation_passes_validated_import_after_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    mode: str,
) -> None:
    """Each local mode selects its finite process supervisor after import validation and consent."""
    path = configuration(tmp_path, mode=mode)
    imported = tmp_path / "examples.jsonl"
    imported.write_text(example().model_dump_json() + "\n")
    observed: list[str] = []

    def execute(launch: RunLaunchConfiguration) -> RunReport:
        """Observe the boundary without downloading weights or entering a runtime."""
        assert launch.import_examples_path == imported
        assert launch.run.mode == mode
        observed.append("execute")
        return report()

    def selected(name: str) -> SimpleNamespace:
        """Verify the selected backend owns the entire learner's finite process lifetime."""
        assert name == "exp.optimize.claas.backends.local.hosting"
        observed.append("import")
        return SimpleNamespace(run_local=execute)

    monkeypatch.setattr(commands.importlib, "import_module", selected)
    result = CliRunner().invoke(
        commands.claas_app,
        [
            command,
            str(path),
            "--import",
            str(imported),
            "--yes",
            "--root",
            str(tmp_path / "budget"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert observed == ["import", "execute"]
    assert "Operator compute reservation" in result.output


def test_modal_invocation_rewrites_only_import_transport_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal receives the bounded local file and its exact run-specific mounted destination."""
    path = configuration(tmp_path)
    launch = RunLaunchConfiguration.model_validate_json(path.read_text()).model_copy(
        update={
            "directory": Path("/state/app"),
            "persistence": "modal-volume",
            "runtime": ResidentVerlSettings(
                checkpoint_root=Path("/state/app/checkpoints"), maximum_output_tokens=4
            ),
        }
    )
    path.write_text(launch.model_dump_json())
    resources = tmp_path / "modal.json"
    resources.write_text(
        '{"app_name":"app","environment_name":"main","volume_name":"volume",'
        '"image_id":"im-fixture","gpu":"L40S"}'
    )
    imported = tmp_path / "examples.jsonl"
    imported.write_text(example().model_dump_json() + "\n")
    called = asyncio.Event()

    async def run_modal(
        launch: RunLaunchConfiguration,
        _resources: ModalLaunch,
        *,
        run_id: str,
        import_path: Path | None,
        report_path: Path,
        console: Console,
    ) -> None:
        """Inspect invocation wiring without creating a Modal client or provider resource."""
        assert launch.import_examples_path == Path("/state/imports/sample.jsonl")
        assert import_path == imported
        assert run_id == "sample"
        assert report_path.name == "sample.json"
        assert isinstance(console, Console)
        called.set()

    monkeypatch.setattr(commands, "run_modal", run_modal)
    result = CliRunner().invoke(
        commands.claas_app,
        [
            "burst",
            str(path),
            "--modal",
            str(resources),
            "--import",
            str(imported),
            "--run-id",
            "sample",
            "--yes",
            "--root",
            str(tmp_path / "budget"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert called.is_set()


def test_failed_local_run_has_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned failure receipt must not look like a successful CLI training run."""
    path = configuration(tmp_path)
    failed = report().model_copy(
        update={
            "status": report().status.model_copy(
                update={"state": "failed", "failure_type": "ValueError"}
            )
        }
    )
    monkeypatch.setattr(
        commands.importlib,
        "import_module",
        lambda _name: SimpleNamespace(run_local=lambda _configuration: failed),
    )
    result = CliRunner().invoke(
        commands.claas_app,
        [
            "burst",
            str(path),
            "--yes",
            "--root",
            str(tmp_path / "budget"),
        ],
    )
    assert result.exit_code == 1
    assert "failed" in result.output


@pytest.mark.parametrize("cost,match", [(0, "positive compute_reservation"), (1, "/state")])
def test_modal_validation_precedes_consent_and_backend_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cost: float,
    match: str,
) -> None:
    """Unpriced compute or an invalid remote mount cannot reach the spend or provider boundary."""
    path = configuration(tmp_path, cost=cost)
    resources = tmp_path / "modal.json"
    resources.write_text(
        '{"app_name":"app","environment_name":"main","volume_name":"volume",'
        '"image_id":"im-fixture","gpu":"L40S"}'
    )

    def forbidden(_name: str) -> None:
        """Fail if invalid remote configuration reaches a backend plugin."""
        pytest.fail("invalid Modal configuration reached backend selection")

    monkeypatch.setattr(commands.importlib, "import_module", forbidden)
    result = CliRunner().invoke(
        commands.claas_app,
        [
            "burst",
            str(path),
            "--modal",
            str(resources),
            "--yes",
            "--root",
            str(tmp_path / "budget"),
        ],
    )
    assert result.exit_code == 2
    assert match in result.output
    assert "Operator compute reservation" not in result.output


@pytest.mark.parametrize("command,mode", [("burst", "burst"), ("serve", "run")])
def test_explicit_zero_cost_local_hardware_uses_shared_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    mode: str,
) -> None:
    """An explicit nonbillable estimate may use existing local hardware without a paid prompt."""
    path = configuration(tmp_path, mode=mode, cost=0)
    imported = tmp_path / "examples.jsonl"
    imported.write_text(example().model_dump_json() + "\n")
    set_maximum_command_cost_usd(1, tmp_path / "budget")
    launches: list[RunLaunchConfiguration] = []

    def execute(launch: RunLaunchConfiguration) -> RunReport:
        """Record validated local ownership without initializing a GPU or allocating a host."""
        launches.append(launch)
        return report()

    def selected(name: str) -> SimpleNamespace:
        """Require the local process supervisor, whose resources already exist on this host."""
        assert name == "exp.optimize.claas.backends.local.hosting"
        return SimpleNamespace(run_local=execute)

    monkeypatch.setattr(commands.importlib, "import_module", selected)
    result = CliRunner().invoke(
        commands.claas_app,
        [command, str(path), "--import", str(imported), "--root", str(tmp_path / "budget")],
    )
    assert result.exit_code == 0, result.output
    assert len(launches) == 1
    assert launches[0].compute_reservation_usd == 0
    assert launches[0].run.mode == mode
    assert launches[0].import_examples_path == imported
    assert "Operator compute reservation: $0.00" in result.output


def test_declared_modal_import_requires_transport_before_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mounted import declaration cannot allocate compute without an uploaded local file."""
    path = configuration(tmp_path)
    launch = RunLaunchConfiguration.model_validate_json(path.read_text()).model_copy(
        update={
            "directory": Path("/state/app"),
            "persistence": "modal-volume",
            "import_examples_path": Path("/state/imports/sample.jsonl"),
            "runtime": ResidentVerlSettings(
                checkpoint_root=Path("/state/app/checkpoints"), maximum_output_tokens=4
            ),
        }
    )
    path.write_text(launch.model_dump_json())
    resources = tmp_path / "modal.json"
    resources.write_text(
        '{"app_name":"app","environment_name":"main","volume_name":"volume",'
        '"image_id":"im-fixture","gpu":"L40S"}'
    )

    def forbidden(
        _console: Console,
        *,
        root: Path,
        yes: bool,
        estimated_cost_usd: float,
        command: str,
    ) -> bool:
        """Fail before consent can authorize a container missing its declared import."""
        pytest.fail("missing Modal import reached spend consent")

    monkeypatch.setattr(commands, "require_spend_consent", forbidden)
    result = CliRunner().invoke(
        commands.claas_app,
        ["burst", str(path), "--modal", str(resources), "--run-id", "sample", "--yes"],
    )
    assert result.exit_code == 2
    assert "declared Modal import requires a local import_path" in result.output
    assert "Operator compute reservation" not in result.output

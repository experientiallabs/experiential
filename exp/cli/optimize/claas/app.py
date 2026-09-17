"""Bounded local or Modal learning commands with provider-free cost preflight."""

import asyncio
import importlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast
from uuid import uuid4

import typer
from rich.console import Console

from exp.cli.optimize.claas.remote import run_modal
from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.optimize.claas.backends.modal.configuration import ModalLaunch
from exp.optimize.claas.backends.modal.validation import validate_launch
from exp.optimize.claas.service.configuration import RunReport
from exp.optimize.claas.service.files import (
    MAXIMUM_CONFIGURATION_BYTES,
    load_configuration,
    load_examples,
    read_regular_file,
)
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration

claas_app = typer.Typer(
    help="Finite learning bursts and resident response/feedback runs.", no_args_is_help=True
)
_console = Console(theme=EXP_THEME)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_CONFIG_ARGUMENT = typer.Argument(..., metavar="CONFIG", help="JSON learning-run configuration.")
_MODAL_OPTION = typer.Option(
    None, "--modal", metavar="MODAL_CONFIG", help="Use this Modal resource configuration."
)
_IMPORT_OPTION = typer.Option(
    None, "--import", metavar="JSONL", help="Import exact training examples before starting."
)


def burst(
    config: Path = _CONFIG_ARGUMENT,
    modal: Path | None = _MODAL_OPTION,
    import_path: Path | None = _IMPORT_OPTION,
    run_id: str | None = typer.Option(
        None, "--run-id", help="Modal run and downloaded receipt identity; generated when omitted."
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm an in-budget operator estimate."),
    root: Path = ROOT_OPTION,
) -> None:
    """Drain the durable experience queue within a finite optimizer-only run."""
    _launch(config, "burst", modal, import_path, run_id, yes, root)


def serve(
    config: Path = _CONFIG_ARGUMENT,
    modal: Path | None = _MODAL_OPTION,
    import_path: Path | None = _IMPORT_OPTION,
    run_id: str | None = typer.Option(
        None, "--run-id", help="Modal run and downloaded receipt identity; generated when omitted."
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm an in-budget operator estimate."),
    root: Path = ROOT_OPTION,
) -> None:
    """Serve responses and feedback while resident engines alternate generation and updates."""
    _launch(config, "run", modal, import_path, run_id, yes, root)


def _launch(
    config_path: Path,
    mode: str,
    modal_path: Path | None,
    import_path: Path | None,
    run_id: str | None,
    yes: bool,
    root: Path,
) -> None:
    """Complete local validation and shared consent before selecting any compute implementation."""
    with usage_error(ValueError, OSError, ImportError):
        configuration = load_configuration(config_path)
        if configuration.run.mode != mode:
            raise ValueError(f"command requires configuration.run.mode='{mode}'")
        identifier = run_id if run_id is not None else "run-" + uuid4().hex
        if _RUN_ID.fullmatch(identifier) is None:
            raise ValueError(
                "run_id must contain 1-64 letters, digits, dots, underscores, or hyphens"
            )
        resources = (
            ModalLaunch.model_validate_json(
                read_regular_file(modal_path, MAXIMUM_CONFIGURATION_BYTES)
            )
            if modal_path is not None
            else None
        )
        if resources is None and configuration.persistence != "local":
            raise ValueError(
                "local execution requires persistence='local'; select --modal for a Volume"
            )
        if import_path is not None:
            local_import = configuration.model_copy(
                update={"import_examples_path": import_path.absolute()}
            )
            load_examples(local_import)
            destination = (
                import_path.absolute()
                if resources is None
                else Path(f"/state/imports/{identifier}.jsonl")
            )
            configuration = configuration.model_copy(update={"import_examples_path": destination})
        elif resources is None:
            load_examples(configuration)
        if resources is not None:
            validate_launch(resources, configuration, identifier, import_path)
        report_path = root / "claas" / "reports" / f"{identifier}.json"
        if resources is not None and report_path.exists():
            raise ValueError("run report already exists; choose another --run-id")
        command = "exp optimize claas " + ("serve" if mode == "run" else "burst")
        _console.print(
            f"Operator compute reservation: ${configuration.compute_reservation_usd:.2f}",
            markup=False,
        )
        if not require_spend_consent(
            _console,
            root=root,
            yes=yes,
            estimated_cost_usd=configuration.compute_reservation_usd,
            command=command,
        ):
            return
        if resources is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            asyncio.run(
                run_modal(
                    configuration,
                    resources,
                    run_id=identifier,
                    import_path=import_path,
                    report_path=report_path,
                    console=_console,
                )
            )
        else:
            if mode == "run":
                _console.print(
                    f"Configured service URL: http://{configuration.host}:{configuration.port}",
                    markup=False,
                )
            module = importlib.import_module("exp.optimize.claas.backends.local.hosting")
            execute = cast(Callable[[RunLaunchConfiguration], RunReport], module.run_local)
            report = execute(configuration)
            _console.print(
                f"Learning run: {report.status.state}; updates: {report.status.updates}",
                markup=False,
            )
            _console.print(
                f"Run report: {configuration.directory / 'run-report.json'}", markup=False
            )
            if report.status.state == "failed":
                raise typer.Exit(1)


claas_app.command("burst")(burst)
claas_app.command("serve")(serve)

"""How a configured build stops: the exact command to run again, and readable paid-work failures."""

from __future__ import annotations

import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console

from exp.cli.shared.consent import require_spend_consent
from exp.common.models import ModelCatalog
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.engines.text.errors import SimulationContentionError


def rerun_command(
    project: str, trace_file: Path | None, import_id: str | None, source: str, identity: str | None
) -> str:
    """The exact automated build command, with every input that selects its traces."""
    if import_id is not None:
        return shlex.join(["exp", "build", project, "--import-id", import_id])
    identity_flags = [] if identity is None else ["--identity", identity]
    return shlex.join(
        ["exp", "build", project, "-t", str(trace_file), "--source", source, *identity_flags]
    )


def require_learning_consent(
    console: Console, catalog: ModelCatalog, *, root: Path, command: str
) -> None:
    """Refuse automation without --yes before indexes are bought, when learning will run.

    Learning's spend is undefined, so it always needs --yes outside a terminal; asking only
    after the indexes are paid for would leave a half-finished build and exit 2.
    """
    from exp.cli.build.learning import resolve_memory_recall

    if resolve_memory_recall(catalog) is not None:
        require_spend_consent(
            console,
            root=root,
            yes=False,
            estimated_cost_usd=None,
            cost_is_upper_bound=False,
            command=command,
            non_interactive=True,
        )


@contextmanager
def paid_work_stops(console: Console) -> Iterator[None]:
    """Stop a failed provider request or contended paid work readably, with exit 1."""
    try:
        yield
    except ProviderTransportError as exc:
        console.print(f"[red]error[/red] a provider request failed: {exc}")
        console.print(
            "Completed embedding batches and finished steps are saved. Run the same exp build "
            "command again to resume; an interrupted request may be retried."
        )
        raise typer.Exit(code=1) from exc
    except SimulationContentionError as exc:
        console.print(f"[red]error[/red] {exc}")
        console.print(
            "Another process owns in-flight paid work. Completed paid work is saved; "
            "run the same exp build command again once that process finishes."
        )
        raise typer.Exit(code=1) from exc

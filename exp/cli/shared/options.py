"""Shared Typer option defaults and the usage-error boundary for exp commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

from exp.common.config import ARTIFACT_DIR

ROOT_OPTION = typer.Option(Path(ARTIFACT_DIR), "--root", help="Local .exp project root.")


@contextmanager
def usage_error(*exception_types: type[Exception]) -> Iterator[None]:
    """Convert the listed domain errors raised in the body into ``typer.BadParameter``."""
    try:
        yield
    except exception_types as exc:
        raise typer.BadParameter(str(exc)) from None

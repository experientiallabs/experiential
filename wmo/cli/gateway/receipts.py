"""Stable human and JSON presentation for gateway management operations."""

from __future__ import annotations

import json

import typer
from pydantic import Field
from rich.console import Console

from wmo.common.core.artifacts import ContractModel, JsonObject

_console = Console()


class GatewayReceipt(ContractModel):
    """One versioned content-free management result."""

    schema_version: int = Field(default=1, frozen=True)
    operation: str
    resource_kind: str
    resource_id: str | None = None
    changed: bool | None = None
    data: JsonObject = Field(default_factory=dict)


class GatewayListReceipt(ContractModel):
    """One versioned content-free management collection."""

    schema_version: int = Field(default=1, frozen=True)
    resource_kind: str
    items: tuple[JsonObject, ...]


def emit_receipt(receipt: GatewayReceipt, *, json_output: bool, human: str) -> None:
    """Write exactly one receipt document or one human-facing summary.

    Args:
        receipt: Versioned secret-safe operation result.
        json_output: Whether stdout must contain JSON only.
        human: Concise operator-facing summary.
    """
    if json_output:
        typer.echo(json.dumps(receipt.model_dump(mode="json"), separators=(",", ":")))
        return
    _console.print(human)


def emit_items(
    resource_kind: str,
    items: tuple[ContractModel, ...],
    *,
    json_output: bool,
) -> None:
    """Write a stable resource list without mixing diagnostics into JSON stdout.

    Args:
        resource_kind: Stable collection name.
        items: Typed content-free resource views.
        json_output: Whether stdout must contain JSON only.
    """
    if json_output:
        receipt = GatewayListReceipt(
            resource_kind=resource_kind,
            items=tuple(item.model_dump(mode="json") for item in items),
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), separators=(",", ":")))
        return
    if not items:
        _console.print(f"no {resource_kind}")
        return
    for item in items:
        values = item.model_dump(mode="json", exclude_none=True)
        _console.print(" ".join(f"{name}={value}" for name, value in values.items()))

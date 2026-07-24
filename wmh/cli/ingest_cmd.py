"""`wmh ingest`: normalize traces from any source into OTel-GenAI JSONL, with live progress.

The standalone half of what `wmh build` does first: pick a source (auto-detected for files),
normalize to the harness's OTel span JSONL, and report progress. The output feeds any downstream
command (`wmh build --source otel-genai --file <out>`), and the same event stream drives the
platform's SSE trace-ingest endpoint (`wmh.ingest.stream.ingest_events`), so the CLI and the
hosted flow can't drift.

Output modes: a rich progress bar on a TTY, and `--json` for one D-INGEST event object per line
(machine-readable; what a wrapping process pipes to a wire).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

from wmh.ingest import VendorPull, list_adapters
from wmh.ingest.stream import (
    DetectedEvent,
    DoneEvent,
    ErrorEvent,
    IngestEvent,
    ProgressEvent,
    event_json,
    ingest_events,
)

_console = Console()


def _default_out(file: str | None, source: str | None) -> Path:
    if file is not None:
        return Path(f"{Path(file).stem}.otel.jsonl")
    return Path(f"{source or 'traces'}.otel.jsonl")


def _render_rich(stream: Iterator[IngestEvent]) -> bool:
    """Consume the event stream behind a progress bar; returns True on success."""
    ok = False
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} traces"),
        console=_console,
    ) as progress:
        task_id = None
        for event in stream:
            if isinstance(event, DetectedEvent):
                _console.print(f"detected [bold]{event.format}[/bold] · {event.traces} traces")
                task_id = progress.add_task("normalizing", total=event.traces)
            elif isinstance(event, ProgressEvent) and task_id is not None:
                progress.update(task_id, completed=event.normalized)
                if event.note:
                    progress.update(task_id, description=event.note)
            elif isinstance(event, DoneEvent):
                ok = True
                progress.stop()
                _console.print(
                    f"[green]done[/green] · {event.traces} traces / {event.steps} steps "
                    f"→ [bold]{event.otel_object}[/bold]"
                )
                _console.print(
                    f"build from it: wmh build --name <name> --source otel-genai "
                    f"--file {event.otel_object}"
                )
            elif isinstance(event, ErrorEvent):
                progress.stop()
                code = f" [{event.code}]" if event.code else ""
                _console.print(f"[red]error{code}[/red] {event.message}")
    return ok


def ingest(
    file: str = typer.Option(
        None, "--file", help="Path to an exported traces file (format auto-detected)."
    ),
    source: str = typer.Option(
        None,
        "--source",
        help="Trace source adapter (default: auto-detect from the file). One of: "
        "otel-genai, chat-json, braintrust, phoenix, langfuse, langsmith, posthog, mastra, "
        "postgres.",
    ),
    pull: bool = typer.Option(
        False, "--pull", help="Pull traces live from the source's API (instead of --file)."
    ),
    project: str = typer.Option(None, "--project", help="Vendor project/workspace id (--pull)."),
    api_key: str = typer.Option(None, "--api-key", help="Vendor API key (else env var)."),
    since: str = typer.Option(None, "--since", help="Only pull traces since this ISO timestamp."),
    limit: int = typer.Option(None, "--limit", help="Max number of traces to ingest."),
    dsn: str = typer.Option(
        None, "--dsn", help="Postgres connection string (implies --source postgres)."
    ),
    table: str = typer.Option(None, "--table", help="Postgres table holding the trace rows."),
    trace_id_column: str = typer.Option(
        None, "--trace-id-column", help="Postgres column grouping rows into traces."
    ),
    payload_column: str = typer.Option(
        None, "--payload-column", help="Postgres JSON column holding the trace payloads."
    ),
    order_column: str = typer.Option(
        None, "--order-column", help="Postgres column ordering rows (default: created_at)."
    ),
    out: str = typer.Option(
        None, "--out", help="Output OTel JSONL path (default: <input>.otel.jsonl)."
    ),
    json_events: bool = typer.Option(
        False, "--json", help="Emit one JSON progress event per line instead of the rich UI."
    ),
) -> None:
    """Normalize traces from a file, vendor API, or Postgres table into OTel JSONL.

    No model is built: the output corpus feeds `wmh build --source otel-genai --file <out>` (or
    any other command that reads a trace file). Progress events follow the D-INGEST vocabulary:
    detected -> progress... -> done | error.
    """
    if (dsn is not None or table is not None) and source is None:
        source = "postgres"
    is_pull = pull or dsn is not None or table is not None
    if file is None and not is_pull:
        raise typer.BadParameter(
            "provide --file <export>, --pull (with --source), or --dsn/--table for postgres"
        )
    if file is not None and is_pull:
        raise typer.BadParameter("pass either --file or a pull (--pull/--dsn/--table), not both")
    if source is not None and source not in list_adapters():
        raise typer.BadParameter(
            f"unknown --source {source!r}; choose one of: {', '.join(list_adapters())}"
        )
    out_path = Path(out) if out else _default_out(file, source)
    vendor_pull = (
        VendorPull(
            api_key=api_key,
            project=project,
            since=since,
            limit=limit,
            dsn=dsn,
            table=table,
            trace_id_column=trace_id_column,
            payload_column=payload_column,
            order_column=order_column,
        )
        if is_pull
        else None
    )
    stream = ingest_events(file=file, pull=vendor_pull, source=source, out=out_path, limit=limit)

    if json_events:
        ok = False
        for event in stream:
            # soft_wrap: machine-readable lines must never be broken at terminal width.
            _console.print(
                json.dumps(event_json(event)), markup=False, highlight=False, soft_wrap=True
            )
            ok = isinstance(event, DoneEvent) or (ok and not isinstance(event, ErrorEvent))
    else:
        ok = _render_rich(stream)
    if not ok:
        raise typer.Exit(code=1)

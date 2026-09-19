"""Own the foreground capture lifetime in its supported Python interpreter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import socket
import subprocess
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.live import Live
from rich.text import Text

from exp.cli.capture.auth import CaptureCredentials, capture_credentials
from exp.cli.capture.session import run_session
from exp.cli.shared.theme import EXP_THEME
from exp.common.auth.paths import provider_data_dir
from exp.runtime.capture.certificates import (
    certificate_is_trusted,
    prepare_certificate,
    trust_certificate,
)
from exp.runtime.capture.control import CaptureCloudError, CaptureRun, CaptureRunClient
from exp.runtime.capture.upload import CaptureUploader, UploadStats


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the internal process entrypoint with the public CLI's validated arguments."""
    parser = argparse.ArgumentParser(description="Experiential foreground capture runner")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--domain", action="append", required=True)
    args = parser.parse_args(arguments)
    console = Console(theme=EXP_THEME)
    try:
        _capture(console, domains=tuple(args.domain), root=args.root)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f"Capture could not continue: {exc}", markup=False)
        console.print("If provider requests are failing, run exp capture reset.")
        return 1
    return 0


def _capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
    """Prepare normal authentication and certificates, then own one foreground run."""
    credentials = capture_credentials(console=console, environment=os.environ, root=root)
    asyncio.run(_capture_authenticated(console, domains=domains, credentials=credentials))


async def _capture_authenticated(
    console: Console,
    *,
    domains: tuple[str, ...],
    credentials: CaptureCredentials,
) -> None:
    """Keep all cancellable control-plane requests in the foreground event loop."""
    control = CaptureRunClient(base_url=credentials.api_url, api_key=credentials.api_key)
    run: CaptureRun | None = None
    uploader: CaptureUploader | None = None
    live = Live(console=console, auto_refresh=False)
    try:
        organization = await control.whoami()
        run = await control.start(
            organization.org_id,
            uuid4(),
            label=socket.gethostname(),
            version=version("experiential"),
        )
        console.print(f"Organization: {organization.org_name}", markup=False)
        console.print(f"Provider domains: {', '.join(domains)}", markup=False)
        console.print("Captures model prompts, responses, and tool content from these domains.")
        console.print("macOS administrator authorization is needed for temporary routing.")
        data_dir = provider_data_dir() / "capture"
        ca_directory = data_dir / "ca"
        origin_namespace = hashlib.sha256(credentials.api_url.encode()).hexdigest()[:16]
        uploader = CaptureUploader(
            base_url=credentials.api_url,
            org_id=str(organization.org_id),
            run_id=str(run.id),
            api_key=credentials.api_key,
            upload_origin=run.upload_origin,
            upload_path_prefix=run.upload_path_prefix,
            spool_dir=data_dir
            / "spool"
            / origin_namespace
            / str(organization.org_id)
            / str(run.id),
        )
        certificate = prepare_certificate(ca_directory)
        if not certificate_is_trusted(certificate, domains=domains):
            console.print("First-time setup: trust Capture's certificate for your macOS user.")
            trust_certificate(certificate, domains=domains)
        console.print(
            "If a client reports a certificate error, stop capture and configure its CA trust."
        )

        def started() -> None:
            """Show active capture only after both proxy and hosts redirection are ready."""
            console.print(
                "[green]Capturing.[/green] Use your AI apps normally. Press Ctrl+C to stop."
            )
            console.print(
                f"Dashboard: {credentials.web_url}/api-keys?section=capture", markup=False
            )
            live.start()

        def progress(stats: UploadStats) -> None:
            """Render counters without displaying prompts or provider credentials."""
            live.update(
                Text(
                    f"{stats.captured_exchanges} requests captured"
                    f" · {stats.uploaded_batches} batches uploaded"
                    f" · {stats.pending_batches} pending · {stats.dropped_exchanges} dropped"
                    f" · {stats.upload_errors} upload errors"
                ),
                refresh=True,
            )

        def warning(message: str) -> None:
            """Show a content-free recovery message alongside live counters."""
            console.print(message, markup=False)

        stats = await run_session(
            domains=domains,
            ca_directory=ca_directory,
            uploader=uploader,
            control=control,
            run=run,
            on_started=started,
            on_progress=progress,
            on_warning=warning,
        )
        progress(stats)
        live.stop()
        console.print("[green]Capture stopped. Networking restored.[/green]")
        if stats.pending_batches:
            console.print(
                f"{stats.pending_batches} batches pending; the next exp capture retries them."
            )
    finally:
        live.stop()
        if run is not None:
            stats = uploader.stats if uploader is not None else None
            try:
                await control.end(
                    run,
                    pending_batches=uploader.pending_current_run if uploader else 0,
                    upload_errors=stats.upload_errors if stats else 0,
                )
            except CaptureCloudError:
                console.print(
                    "Platform could not record the stop; this run will show as disconnected."
                )
        await control.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Own the foreground capture lifetime in its supported Python interpreter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import socket
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from mitmproxy_rs.local import LocalRedirector
from rich.console import Console

from exp.cli.capture.approval import open_pending_approval_settings
from exp.cli.capture.auth import CaptureCredentials, capture_credentials
from exp.cli.capture.display import CaptureDisplay
from exp.cli.capture.session import run_session
from exp.cli.shared.theme import EXP_THEME
from exp.common.auth.paths import provider_data_dir
from exp.runtime.capture.certificates import (
    capture_certificate_directory,
    certificate_is_trusted,
    prepare_certificate,
    trust_certificate,
)
from exp.runtime.capture.control import CaptureCloudError, CaptureRun, CaptureRunClient
from exp.runtime.capture.local_backend import capture_instance, require_local_backend
from exp.runtime.capture.upload import CaptureUploader


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the internal process entrypoint with the public CLI's validated arguments."""
    parser = argparse.ArgumentParser(description="Experiential foreground capture runner")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--domain", action="append", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(arguments)
    console = Console(theme=EXP_THEME)
    try:
        _capture(console, domains=tuple(args.domain), root=args.root, verbose=args.verbose)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f"Capture could not continue: {exc}", markup=False)
        return 1
    return 0


def _capture(
    console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool = False
) -> None:
    """Prepare normal authentication and certificates, then own one foreground run."""
    with console.status("Checking Capture…"):
        require_local_backend(installation_is_current=_installed_bundle_matches)
    with capture_instance():
        credentials = capture_credentials(console=console, environment=os.environ, root=root)
        asyncio.run(
            _capture_authenticated(
                console, domains=domains, credentials=credentials, verbose=verbose
            )
        )


def _installed_bundle_matches() -> bool:
    """Query the native installer's app identity only on its supported platform."""
    if sys.platform != "darwin":
        raise RuntimeError("System Capture currently supports macOS only.")
    return LocalRedirector.installation_is_current()


async def _capture_authenticated(
    console: Console,
    *,
    domains: tuple[str, ...],
    credentials: CaptureCredentials,
    verbose: bool = False,
) -> None:
    """Keep all cancellable control-plane requests in the foreground event loop."""
    control = CaptureRunClient(base_url=credentials.api_url, api_key=credentials.api_key)
    run: CaptureRun | None = None
    uploader: CaptureUploader | None = None
    display = CaptureDisplay(console, verbose=verbose)
    approval_task: asyncio.Task[bool] | None = None
    try:
        display.phase("Connecting to Experiential…")
        organization = await control.whoami()
        run = await control.start(
            organization.org_id,
            uuid4(),
            label=socket.gethostname(),
            version=version("experiential"),
        )
        if verbose:
            console.print(f"Organization: {organization.org_name}", markup=False)
            console.print(f"Provider domains: {', '.join(domains)}", markup=False)
            console.print("Captures model prompts, responses, and tool content from these domains.")
            console.print(
                "Approve Mitmproxy Redirector if macOS requests network extension access."
            )
            console.print(
                "Other traffic passes through without capture. DNS settings stay unchanged."
            )
        data_dir = provider_data_dir() / "capture"
        ca_directory = capture_certificate_directory(data_dir, domains)
        origin_namespace = hashlib.sha256(credentials.api_url.encode()).hexdigest()[:16]
        uploader = CaptureUploader(
            base_url=credentials.api_url,
            org_id=str(organization.org_id),
            run_id=str(run.id),
            api_key=credentials.api_key,
            upload_origin=run.upload_origin,
            upload_path_prefix=run.upload_path_prefix,
            on_diagnostic=display.diagnostic if verbose else None,
            spool_dir=data_dir
            / "spool"
            / origin_namespace
            / str(organization.org_id)
            / str(run.id),
        )
        display.phase("Checking certificate…")
        certificate = prepare_certificate(ca_directory, domains)
        if not certificate_is_trusted(certificate, domains=domains):
            display.close()
            console.print("Waiting for certificate approval…")
            if verbose:
                console.print(
                    "First-time setup: trust Capture's certificate for your macOS user. "
                    "The certificate is restricted to the selected provider hosts."
                )
            trust_certificate(certificate, domains=domains)
        if verbose:
            console.print(
                "Clients rejecting Capture's certificate pass through without capture. "
                "Clients with a custom trust store may need the public CA certificate below."
            )
            console.print(f"Public CA: {certificate}", markup=False)
        display.phase("Starting network extension… Ctrl+C to cancel")

        def started() -> None:
            """Show active capture only after network interception is ready."""
            if approval_task is not None:
                approval_task.cancel()
            display.started(
                organization=organization.org_name,
                telemetry_url=f"{credentials.web_url}/api-keys?section=capture",
            )

        def waiting() -> None:
            """Keep the approval hint visible and navigate without blocking startup."""
            nonlocal approval_task
            display.waiting()
            approval_task = asyncio.create_task(open_pending_approval_settings())

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
            on_progress=display.progress,
            on_warning=warning,
            on_waiting=waiting,
            on_bypass=display.bypassed,
            on_diagnostic=display.diagnostic if verbose else None,
        )
        display.stopped(stats)
    finally:
        if approval_task is not None:
            approval_task.cancel()
            await asyncio.gather(approval_task, return_exceptions=True)
        display.close()
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

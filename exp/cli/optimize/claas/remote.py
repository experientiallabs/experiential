"""Foreground ownership and receipt download for one explicitly selected Modal run."""

import asyncio
import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.console import Console

from exp.common.core.artifacts import ContractModel
from exp.optimize.claas.backends.modal.configuration import ModalLaunch
from exp.optimize.claas.service.files import write_contract
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration

if TYPE_CHECKING:
    from exp.optimize.claas.backends.modal.hosting import ModalRunHandle

logger = logging.getLogger(__name__)


_CANCELLATION_JOIN_SECONDS = 30.0
_STOP_RPC_OVERHEAD_SECONDS = 60.0


class RecoveryReceipt(ContractModel):
    """Retain exact remote identity when cleanup or report transport cannot be confirmed."""

    sandbox_id: str
    app_name: str
    environment_name: str
    volume_name: str
    run_id: str
    report_source: str
    state: Literal["running", "exited", "stopped", "unknown"]
    failure_type: str | None = None


def _observe_cleanup(task: asyncio.Task[None]) -> None:
    """Retrieve a late cleanup failure after recording an explicitly uncertain remote outcome."""
    if not task.cancelled():
        task.exception()


async def _wait_cleanup(task: asyncio.Task[None], deadline: float) -> asyncio.CancelledError | None:
    """Wait against one monotonic deadline without extending it on repeated caller cancellation."""
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError as error:
            cancelled = cancelled or error
    return cancelled


async def _join_cleanup(task: asyncio.Task[None], *, timeout_seconds: float) -> None:
    """Bound stop and cancellation joins while preserving the caller's interruption signal.

    Modal RPC cancellation is cooperative. The second finite join covers its termination
    fallback; an unconfirmed outcome retains a recovery receipt and the Sandbox hard lifetime.
    """
    cancelled = await _wait_cleanup(task, asyncio.get_running_loop().time() + timeout_seconds)
    expired = not task.done()
    if expired:
        task.cancel()
        later = await _wait_cleanup(
            task, asyncio.get_running_loop().time() + _CANCELLATION_JOIN_SECONDS
        )
        cancelled = cancelled or later
    if not task.done():
        task.cancel()
        task.add_done_callback(_observe_cleanup)
    if cancelled is not None:
        failure = task.exception() if task.done() and not task.cancelled() else None
        if expired:
            raise cancelled from TimeoutError("Modal cleanup exceeded its foreground deadline")
        if failure is not None:
            raise cancelled from failure
        raise cancelled
    if expired:
        if task.done() and not task.cancelled():
            task.exception()
        raise TimeoutError("Modal cleanup exceeded its foreground deadline; retain the Sandbox ID")
    task.result()


async def _download(handle: "ModalRunHandle", source: str, target: Path, console: Console) -> None:
    """Download one terminal receipt without replacing a completed run with a transport error."""
    try:
        async with asyncio.timeout(30):
            await handle.download(source, target)
    except Exception as error:  # noqa: BLE001 - report transport is explicitly best-effort
        logger.warning("Run report download failed: %s", type(error).__name__)
        console.print(f"Report remains in the Modal Volume at {source}.", markup=False)
    else:
        console.print(f"Run report: {target}", markup=False)


async def run_modal(
    configuration: RunLaunchConfiguration,
    resources: ModalLaunch,
    *,
    run_id: str,
    import_path: Path | None,
    report_path: Path,
    console: Console,
) -> None:
    """Start after consent, retain foreground ownership, and stop the exact run on Ctrl-C."""
    recovery_path = report_path.with_suffix(".recovery.json")
    if recovery_path.exists():
        raise ValueError("run recovery receipt already exists; choose another --run-id")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    source = (configuration.directory / "run-report.json").relative_to("/state").as_posix()
    module = importlib.import_module("exp.optimize.claas.backends.modal.hosting")
    launcher = module.ModalLauncher(resources)
    handle = await launcher.start(configuration, run_id=run_id, import_path=import_path)
    receipt = RecoveryReceipt(
        sandbox_id=handle.sandbox_id,
        app_name=resources.app_name,
        environment_name=resources.environment_name,
        volume_name=resources.volume_name,
        run_id=run_id,
        report_source=source,
        state="running",
    )
    try:
        console.print(f"Modal Sandbox: {handle.sandbox_id}", markup=False)
        write_contract(recovery_path, receipt)
        if handle.endpoint is not None:
            console.print(f"Service endpoint: {handle.endpoint}", markup=False)
        await handle.wait()
    except BaseException as original:
        cleanup = asyncio.create_task(handle.stop())
        cleanup_error: BaseException | None = None
        try:
            await _join_cleanup(
                cleanup,
                timeout_seconds=configuration.run.cleanup_timeout_seconds
                + _STOP_RPC_OVERHEAD_SECONDS,
            )
        except BaseException as error:  # noqa: BLE001 - persist outcome before propagating failures
            cleanup_error = error
        stopped = cleanup.done() and not cleanup.cancelled() and cleanup.exception() is None
        receipt = receipt.model_copy(
            update={
                "state": "stopped" if stopped else "unknown",
                "failure_type": type(cleanup_error.__cause__ or cleanup_error).__name__
                if cleanup_error
                else type(original).__name__,
            }
        )
        saved = True
        try:
            write_contract(recovery_path, receipt)
        except OSError as error:
            saved = False
            logger.warning("Could not persist Sandbox recovery receipt: %s", type(error).__name__)
        outcome = "completed" if stopped else "is unconfirmed"
        console.print(
            f"Modal cleanup {outcome} for {handle.sandbox_id}. "
            f"Report remains in Volume {resources.volume_name} at {source}.",
            markup=False,
        )
        if saved:
            console.print(f"Recovery receipt: {recovery_path}", markup=False)
        else:
            console.print(
                "Recovery receipt could not be saved; retain the Sandbox ID.", markup=False
            )
        if isinstance(cleanup_error, asyncio.CancelledError):
            raise cleanup_error from original
        if stopped:
            await _download(handle, source, report_path, console)
        raise
    write_contract(recovery_path, receipt.model_copy(update={"state": "exited"}))
    await _download(handle, source, report_path, console)

"""Finite learning-run execution shared by local processes and infrastructure adapters."""

import asyncio
from collections.abc import Awaitable, Callable

from exp.optimize.claas.service.configuration import RunReport
from exp.optimize.claas.service.contracts import LearnerRuntimeFactory
from exp.optimize.claas.service.controller import LearningController
from exp.optimize.claas.service.files import load_examples, write_contract
from exp.optimize.claas.service.hosting import serve_run
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration


async def execute_run(
    configuration: RunLaunchConfiguration,
    factory: LearnerRuntimeFactory,
    *,
    api_key: str | None = None,
    stop: asyncio.Event | None = None,
    persist: Callable[[], Awaitable[None]] | None = None,
) -> RunReport:
    """Own one runtime, drain on orderly shutdown, and persist its terminal receipt.

    The injected factory keeps compute ownership interchangeable. External environments
    call the HTTP API themselves; this boundary does not execute simulations or deploy.
    """
    if configuration.run.mode == "run" and (api_key is None or len(api_key) < 16):
        raise ValueError("set the configured authentication environment variable to a strong key")
    examples = load_examples(configuration)
    controller = LearningController(
        configuration.directory, configuration.spec, factory, configuration.run, persist=persist
    )
    with controller.hold_directory():
        stopped = stop if stop is not None else asyncio.Event()
        report: RunReport | None = None
        failure: BaseException | None = None
        try:
            write_contract(configuration.directory / "launch.json", configuration)
            if examples:
                await controller.import_examples(examples)
            if configuration.run.mode == "burst":
                report = await controller.drain()
            elif not stopped.is_set():
                await _start(controller, stopped)
                if not stopped.is_set():
                    assert api_key is not None
                    await serve_run(controller, configuration, api_key, stopped)
                if (await controller.status()).state == "running":
                    await controller.drain()
        except BaseException as error:
            failure = error
            raise
        finally:
            cleanup = asyncio.create_task(
                _finish(
                    controller, configuration, persist, type(failure).__name__ if failure else None
                )
            )
            cancelled = failure if isinstance(failure, asyncio.CancelledError) else None
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as error:
                    cancelled = error
                except Exception:  # noqa: BLE001 - retrieve owned cleanup without losing the cause
                    break
            try:
                report = cleanup.result()
            except BaseException as cleanup_error:
                if isinstance(cleanup_error, asyncio.CancelledError):
                    raise
                if cancelled is not None:
                    raise cancelled from cleanup_error
                if failure is not None:
                    raise failure from cleanup_error
                raise
            if cancelled is not None:
                raise cancelled
        return report


async def _start(controller: LearningController, stop: asyncio.Event) -> None:
    """Cancel owned startup when shutdown arrives before a resident engine is ready."""
    startup = asyncio.create_task(controller.start())
    shutdown = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait((startup, shutdown), return_when=asyncio.FIRST_COMPLETED)
        if startup in done:
            startup.result()
        else:
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
    finally:
        shutdown.cancel()
        if not startup.done():
            startup.cancel()
        await asyncio.gather(startup, shutdown, return_exceptions=True)


async def _finish(
    controller: LearningController,
    configuration: RunLaunchConfiguration,
    persist: Callable[[], Awaitable[None]] | None,
    failure: str | None,
) -> RunReport:
    """Join owned runtime cleanup before reporting and committing terminal state."""
    cleanup_failure: BaseException | None = None
    try:
        report = await controller.close(reason="shutdown")
    except BaseException as error:  # noqa: BLE001 - persist failed cleanup before propagating it
        cleanup_failure = error
        report = RunReport(
            status=await controller.status(), checkpoint=controller.buffer.checkpoint()
        )
    if failure is not None:
        report = report.model_copy(
            update={
                "status": report.status.model_copy(
                    update={"state": "failed", "failure_type": failure, "stop_reason": "failed"}
                )
            }
        )
    write_contract(configuration.directory / "run-report.json", report)
    if persist is not None:
        try:
            async with asyncio.timeout(configuration.run.cleanup_timeout_seconds):
                await persist()
        except BaseException as error:
            report = report.model_copy(
                update={
                    "status": report.status.model_copy(
                        update={
                            "state": "failed",
                            "stop_reason": "failed",
                            "failure_type": report.status.failure_type or type(error).__name__,
                            "cleanup_failure_type": type(error).__name__,
                        }
                    )
                }
            )
            write_contract(configuration.directory / "run-report.json", report)
            raise
    if cleanup_failure is not None:
        raise cleanup_failure
    return report

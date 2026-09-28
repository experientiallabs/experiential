"""One HTTP listener for a resident learner with externally owned shutdown signals."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn

from exp.optimize.claas.service.controller import LearningController
from exp.optimize.claas.service.http.app import create_app
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration


class LearningServer(uvicorn.Server):
    """Let the launcher coordinate one process-wide signal and checkpoint boundary."""

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        """Leave signal registration to the owner of the learning runtime."""
        yield


async def serve_run(
    controller: LearningController,
    configuration: RunLaunchConfiguration,
    api_key: str,
    stop: asyncio.Event,
) -> None:
    """Serve until the finite controller closes or the owner requests a graceful stop."""
    app = create_app(controller, api_key=api_key, manage_lifecycle=False)
    server = LearningServer(
        uvicorn.Config(
            app,
            host=configuration.host,
            port=configuration.port,
            access_log=False,
            timeout_graceful_shutdown=int(configuration.run.cleanup_timeout_seconds),
        )
    )

    async def watch() -> None:
        """Close admission when the run expires, fails, drains, or receives a signal."""
        while not stop.is_set():
            if (await controller.status()).state in {"closing", "closed", "failed"}:
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.1)
            except TimeoutError:
                continue
        server.should_exit = True

    watcher = asyncio.create_task(watch(), name="claas-http-lifetime")
    failure: BaseException | None = None
    try:
        try:
            await server.serve()
        except SystemExit as error:
            raise RuntimeError(
                "learner HTTP listener failed to bind the configured address"
            ) from error
        if not server.started:
            raise RuntimeError("learner HTTP listener did not start; check the configured port")
    except BaseException as error:
        failure = error
        raise
    finally:
        cleanup = asyncio.create_task(
            _close_listener(server, configuration, watcher, interrupted=failure is not None)
        )
        cancelled = failure if isinstance(failure, asyncio.CancelledError) else None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:  # noqa: BLE001 - retrieve owned cleanup below
                break
        try:
            cleanup.result()
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


async def _close_listener(
    server: LearningServer,
    configuration: RunLaunchConfiguration,
    watcher: asyncio.Task[None],
    *,
    interrupted: bool,
) -> None:
    """Close HTTP admission and owned requests before runtime cleanup can release compute."""
    server.should_exit = True
    async with asyncio.timeout(configuration.run.cleanup_timeout_seconds):
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        if not interrupted:
            return
        if server.started:
            await server.shutdown()
            await asyncio.gather(*server.server_state.tasks, return_exceptions=True)
        elif hasattr(server, "lifespan"):
            await server.lifespan.shutdown()

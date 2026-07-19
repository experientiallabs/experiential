"""Cancellation-safe local Docker environment for the pinned Harbor adapter."""

from __future__ import annotations

import asyncio

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment

REAPING_DOCKER_ENVIRONMENT_IMPORT_PATH = (
    "wmh.evals.harbor.docker_environment:ReapingDockerEnvironment"
)
_PROCESS_TERMINATE_TIMEOUT_S = 5.0


class ReapingDockerEnvironment(DockerEnvironment):
    """Harbor Docker environment that reaps buffered commands on every exit path.

    Harbor 0.18 terminates a buffered Compose subprocess on its own timeout, but an outer task
    cancellation exits ``communicate`` without stopping the subprocess. WMH uses this explicit
    subclass for local evaluation so trial and job cancellation cannot orphan a Docker CLI process.
    """

    @staticmethod
    async def _collect_buffered_output(
        process: asyncio.subprocess.Process,
        *,
        timeout_sec: int | float | None,
        stdin_data: bytes | None = None,
    ) -> ExecResult:
        try:
            if timeout_sec is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input=stdin_data),
                    timeout=timeout_sec,
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate(input=stdin_data)
        except TimeoutError:
            await _terminate_and_reap(process)
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
        except BaseException:
            await _terminate_and_reap(process)
            raise

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )


async def _terminate_and_reap(process: asyncio.subprocess.Process) -> None:
    """Terminate one subprocess and finish cleanup despite caller cancellation."""

    async def cleanup() -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_PROCESS_TERMINATE_TIMEOUT_S,
            )
            return
        except TimeoutError:
            pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()

    cleanup_task = asyncio.create_task(cleanup())
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - cleanup task outcome is sanitized below
            break
    try:
        await cleanup_task
    except Exception:  # noqa: BLE001 - never expose Compose process details
        raise RuntimeError("Docker command cleanup was not proved") from None

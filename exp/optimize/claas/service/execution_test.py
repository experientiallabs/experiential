"""Drive finite burst and HTTP run lifetimes through real SQLite and sockets."""

import asyncio
import socket
from pathlib import Path

import httpx
import pytest

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.buffer.store_test import item
from exp.optimize.claas.service.configuration import RunConfiguration, RunReport
from exp.optimize.claas.service.contracts import RunMode
from exp.optimize.claas.service.controller import LearningController
from exp.optimize.claas.service.controller_test import Runtime
from exp.optimize.claas.service.execution import execute_run
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration
from exp.optimize.claas.training_contracts import ClaasTrainingSpec, TrainingCheckpoint
from exp.optimize.claas.training_contracts_test import spec


def configuration(
    directory: Path, *, mode: str = "burst", port: int = 8000
) -> RunLaunchConfiguration:
    """Freeze a small CPU fixture recipe with the same public launch model as GPU runs."""
    return RunLaunchConfiguration(
        directory=directory,
        spec=spec().model_copy(update={"max_batch_examples": 2}),
        run=RunConfiguration.model_validate({"mode": mode, "minimum_ready_examples": 2}),
        runtime=ResidentVerlSettings(
            checkpoint_root=directory / "checkpoints", maximum_output_tokens=64
        ),
        compute_reservation_usd=0,
        port=port,
    )


def test_burst_import_two_updates_and_resume_empty(tmp_path: Path) -> None:
    """A launch owns one runtime across two updates and an empty restart acquires no GPU."""
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(item(name).model_dump_json() for name in ("a", "b", "c")))
    config = configuration(tmp_path / "state").model_copy(update={"import_examples_path": source})
    runtime = Runtime()
    report = asyncio.run(execute_run(config, runtime))
    assert report.status.state == "closed"
    assert report.status.buffer.consumed == 3
    assert runtime.optimizations == 2
    assert runtime.open_count == runtime.close_count == 1
    saved = RunReport.model_validate_json((config.directory / "run-report.json").read_bytes())
    assert saved == report
    again = Runtime()
    assert asyncio.run(execute_run(config, again)).status.buffer.consumed == 3
    assert again.open_count == 0


def test_full_run_serves_authenticated_http_and_stops(tmp_path: Path) -> None:
    """Launch a real listener, authenticate a status call, and stop without losing ownership."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    config = configuration(tmp_path, mode="run", port=port)
    runtime = Runtime()

    async def run() -> None:
        """Drive the local public HTTP surface while the owned run is live."""
        stop = asyncio.Event()
        task = asyncio.create_task(
            execute_run(config, runtime, api_key="test-key-at-least-16", stop=stop)
        )
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                for _ in range(100):
                    try:
                        response = await client.get("/v1/status")
                        break
                    except httpx.ConnectError:
                        if task.done():
                            task.result()
                        await asyncio.sleep(0.02)
                else:
                    raise AssertionError("local learning listener did not start")
                assert response.status_code == 401
                response = await client.get(
                    "/v1/status", headers={"Authorization": "Bearer test-key-at-least-16"}
                )
                assert response.status_code == 200
                assert response.json()["state"] == "running"
        finally:
            stop.set()
            report = await asyncio.wait_for(task, 5)
        assert report.status.state == "closed"
        assert runtime.open_count == runtime.close_count == 1

    asyncio.run(run())


def test_invalid_import_and_missing_key_do_not_allocate(tmp_path: Path) -> None:
    """Reject preflight input before acquiring the runtime or creating durable state."""
    runtime = Runtime()
    with pytest.raises(ValueError, match="authentication"):
        asyncio.run(execute_run(configuration(tmp_path / "run", mode="run"), runtime))
    source = tmp_path / "bad.jsonl"
    source.write_text("{}\n")
    with pytest.raises(ValueError):
        asyncio.run(
            execute_run(
                configuration(tmp_path / "burst").model_copy(
                    update={"import_examples_path": source}
                ),
                runtime,
            )
        )
    assert runtime.open_count == 0
    assert not (tmp_path / "burst").exists()


class CleanupRuntime(Runtime):
    """Model a runtime that reports failed cleanup after releasing its awaited test gate."""

    async def close(self) -> None:
        """Fail explicitly instead of pretending uncertain resource cleanup succeeded."""
        await super().close()
        raise OSError("fixture cleanup failure")


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cancelled_http_run_closes_listener_and_preserves_cancellation(
    tmp_path: Path, cleanup_fails: bool
) -> None:
    """Join repeated cancellation through listener cleanup and retain a failed receipt."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    config = configuration(tmp_path, mode="run", port=port)
    runtime = CleanupRuntime() if cleanup_fails else Runtime()

    async def run() -> None:
        runtime.close_gate = asyncio.Event()
        task = asyncio.create_task(execute_run(config, runtime, api_key="test-key-at-least-16"))
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            for _ in range(100):
                try:
                    assert (await client.get("/v1/status")).status_code == 401
                    break
                except httpx.ConnectError:
                    if task.done():
                        task.result()
                    await asyncio.sleep(0.01)
            else:
                raise AssertionError("listener did not start")
        task.cancel("original cancellation")
        await asyncio.wait_for(runtime.close_entered.wait(), 2)
        task.cancel("repeated cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        runtime.close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert runtime.close_count == 1
        saved = RunReport.model_validate_json((tmp_path / "run-report.json").read_bytes())
        assert saved.status.state == "failed"
        assert saved.status.failure_type == "CancelledError"
        assert saved.status.cleanup_failure_type == ("OSError" if cleanup_fails else None)
        with socket.socket() as replacement:
            replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            replacement.bind(("127.0.0.1", port))
            replacement.listen()

    asyncio.run(run())


def test_bind_failure_releases_runtime_and_records_failure(tmp_path: Path) -> None:
    """An occupied port raises a regular failure rather than escaping the loop as SystemExit."""
    runtime = Runtime()
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(RuntimeError, match="bind"):
            asyncio.run(
                execute_run(
                    configuration(tmp_path, mode="run", port=port),
                    runtime,
                    api_key="test-key-at-least-16",
                )
            )
    assert runtime.open_count == runtime.close_count == 1
    saved = RunReport.model_validate_json((tmp_path / "run-report.json").read_bytes())
    assert saved.status.failure_type == "RuntimeError"


class StartingRuntime(Runtime):
    """Own partially allocated fixture resources until asynchronous startup completes."""

    def __init__(self) -> None:
        super().__init__()
        self.open_entered = asyncio.Event()

    async def open(
        self, spec: ClaasTrainingSpec, resume: TrainingCheckpoint | None = None, *, mode: RunMode
    ) -> Runtime:
        await super().open(spec, resume, mode=mode)
        self.open_entered.set()
        try:
            await asyncio.Event().wait()
        except BaseException:
            await self.close()
            raise
        return self


def test_stop_during_startup_joins_partial_runtime_cleanup(tmp_path: Path) -> None:
    """A shutdown signal before HTTP admission cancels and joins factory startup once."""
    runtime = StartingRuntime()

    async def run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(
            execute_run(
                configuration(tmp_path, mode="run"),
                runtime,
                api_key="test-key-at-least-16",
                stop=stop,
            )
        )
        await asyncio.wait_for(runtime.open_entered.wait(), 1)
        stop.set()
        report = await asyncio.wait_for(task, 2)
        assert report.status.failure_type == "CancelledError"
        assert runtime.open_count == runtime.close_count == 1

    asyncio.run(run())


def test_terminal_report_persistence_keeps_directory_owned(tmp_path: Path) -> None:
    """A replacement run cannot overwrite launch or terminal evidence before final persistence."""
    config = configuration(tmp_path)

    async def run() -> None:
        finalizing = asyncio.Event()
        release = asyncio.Event()

        async def persist() -> None:
            if (tmp_path / "run-report.json").exists():
                finalizing.set()
                await release.wait()

        task = asyncio.create_task(execute_run(config, Runtime(), persist=persist))
        await asyncio.wait_for(finalizing.wait(), 1)
        rejected = False
        try:
            try:
                overlapping = LearningController(tmp_path, config.spec, Runtime(), config.run)
            except ValueError as error:
                assert "another learner" in str(error)
                rejected = True
            else:
                await overlapping.close()
        finally:
            release.set()
            await asyncio.wait_for(task, 1)
        assert rejected, "replacement acquired the directory before terminal evidence persisted"
        replacement = LearningController(tmp_path, config.spec, Runtime(), config.run)
        await replacement.close()

    asyncio.run(run())


def test_terminal_persistence_failure_is_not_a_successful_local_receipt(tmp_path: Path) -> None:
    """A failed mount acknowledgement leaves a failed local report and propagates its cause."""
    failure = OSError("fixture persistence failure")
    config = configuration(tmp_path)

    async def persist() -> None:
        if (tmp_path / "run-report.json").exists():
            raise failure

    with pytest.raises(OSError) as raised:
        asyncio.run(execute_run(config, Runtime(), persist=persist))
    assert raised.value is failure
    saved = RunReport.model_validate_json((tmp_path / "run-report.json").read_bytes())
    assert saved.status.state == saved.status.stop_reason == "failed"
    assert saved.status.failure_type == saved.status.cleanup_failure_type == "OSError"
    replacement = LearningController(tmp_path, config.spec, Runtime(), config.run)
    asyncio.run(replacement.close())

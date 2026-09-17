"""Foreground Modal ownership and canceled-wait cleanup without provider resources."""

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from exp.cli.optimize.claas import remote
from exp.cli.optimize.claas.app_test import configuration
from exp.optimize.claas.backends.modal.configuration import ModalLaunch
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration


class Handle:
    """An inert Sandbox handle recording lifecycle and artifact transport calls."""

    sandbox_id = "sb-fixture"
    endpoint = "https://example.invalid"

    def __init__(self) -> None:
        """Hold foreground wait and cleanup behind explicit test-owned events."""
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.stopping = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.downloaded: list[str] = []

    async def wait(self) -> int:
        """Block until successful exit or caller cancellation."""
        self.waiting.set()
        await self.release.wait()
        return 0

    async def stop(self) -> None:
        """Keep cleanup active until the test positively releases it."""
        self.stopping.set()
        await self.cleaned.wait()

    async def download(self, relative_path: str, destination: Path) -> None:
        """Write one inert receipt to prove the requested Volume path and local destination."""
        self.downloaded.append(relative_path)
        destination.write_text("{}")


class Launcher:
    """Inert selected-backend constructor with a retained handle."""

    def __init__(self, handle: Handle) -> None:
        """Keep the fixture's ownership identity."""
        self.handle = handle

    async def start(
        self,
        configuration: RunLaunchConfiguration,
        *,
        run_id: str,
        import_path: Path | None,
    ) -> Handle:
        """Validate invocation identity without allocating any remote resource."""
        assert configuration.directory == Path("/state/learning")
        assert run_id == "run-1" and import_path is None
        return self.handle


def test_foreground_cancellation_joins_modal_stop_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canceling the CLI cannot abandon an allocated Sandbox while cleanup is pending."""
    source = configuration(tmp_path)
    config = RunLaunchConfiguration.model_validate_json(source.read_text()).model_copy(
        update={
            "directory": Path("/state/learning"),
        }
    )
    resources = ModalLaunch(
        app_name="app",
        environment_name="main",
        volume_name="volume",
        image_id="im-fixture",
        gpu="L40S",
    )

    async def exercise() -> None:
        """Cancel foreground wait twice and inspect the retained cleanup task."""
        handle = Handle()
        monkeypatch.setattr(
            remote.importlib,
            "import_module",
            lambda _name: SimpleNamespace(ModalLauncher=lambda _resources: Launcher(handle)),
        )
        task = asyncio.create_task(
            remote.run_modal(
                config,
                resources,
                run_id="run-1",
                import_path=None,
                report_path=tmp_path / "report.json",
                console=Console(file=io.StringIO()),
            )
        )
        await handle.waiting.wait()
        task.cancel()
        await handle.stopping.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        handle.cleaned.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert handle.stopping.is_set()

    asyncio.run(exercise())


def test_successful_foreground_exit_downloads_only_the_named_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact retained Sandbox yields one relative Volume artifact after successful exit."""
    source = configuration(tmp_path)
    config = RunLaunchConfiguration.model_validate_json(source.read_text()).model_copy(
        update={
            "directory": Path("/state/learning"),
        }
    )
    resources = ModalLaunch(
        app_name="app",
        environment_name="main",
        volume_name="volume",
        image_id="im-fixture",
        gpu="L40S",
    )

    async def exercise() -> None:
        """Complete foreground waiting and observe the named report transport."""
        handle = Handle()
        handle.release.set()
        monkeypatch.setattr(
            remote.importlib,
            "import_module",
            lambda _name: SimpleNamespace(ModalLauncher=lambda _resources: Launcher(handle)),
        )
        await remote.run_modal(
            config,
            resources,
            run_id="run-1",
            import_path=None,
            report_path=tmp_path / "report.json",
            console=Console(file=io.StringIO()),
        )
        assert handle.downloaded == ["learning/run-report.json"]
        assert (tmp_path / "report.json").read_text() == "{}"
        assert not handle.stopping.is_set()

    asyncio.run(exercise())


def test_cleanup_cancellation_wins_over_a_later_stop_failure() -> None:
    """Late ordinary cleanup errors cannot replace a caller's cancellation signal."""

    async def exercise() -> None:
        """Cancel the waiter, then fail its still-owned cleanup task."""
        entered, release = asyncio.Event(), asyncio.Event()

        async def cleanup() -> None:
            """Hold cleanup until cancellation is observed, then report a transport failure."""
            entered.set()
            await release.wait()
            raise OSError("late stop transport failure")

        owned = asyncio.create_task(cleanup())
        waiter = asyncio.create_task(remote._join_cleanup(owned, timeout_seconds=1))
        await entered.wait()
        waiter.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert owned.done()

    asyncio.run(exercise())


class HangingHandle(Handle):
    """A cancellation-cooperative Modal transport whose stop RPC never responds."""

    def __init__(self) -> None:
        super().__init__()
        self.cleanup_finished = asyncio.Event()

    async def stop(self) -> None:
        self.stopping.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleanup_finished.set()


def test_cleanup_deadline_preserves_cancellation_and_durable_sandbox_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung stop is canceled, joined, and reported without extending the foreground deadline."""
    source = configuration(tmp_path)
    config = RunLaunchConfiguration.model_validate_json(source.read_text()).model_copy(
        update={
            "directory": Path("/state/learning"),
            "run": RunConfiguration(cleanup_timeout_seconds=0.01),
        }
    )
    resources = ModalLaunch(
        app_name="app",
        environment_name="main",
        volume_name="volume",
        image_id="im-fixture",
        gpu="L40S",
    )
    monkeypatch.setattr(remote, "_STOP_RPC_OVERHEAD_SECONDS", 0.01)
    monkeypatch.setattr(remote, "_CANCELLATION_JOIN_SECONDS", 0.02)

    async def exercise() -> None:
        handle = HangingHandle()
        monkeypatch.setattr(
            remote.importlib,
            "import_module",
            lambda _name: SimpleNamespace(ModalLauncher=lambda _resources: Launcher(handle)),
        )
        output = io.StringIO()
        task = asyncio.create_task(
            remote.run_modal(
                config,
                resources,
                run_id="run-1",
                import_path=None,
                report_path=tmp_path / "report.json",
                console=Console(file=output),
            )
        )
        await handle.waiting.wait()
        initial = remote.RecoveryReceipt.model_validate_json(
            (tmp_path / "report.recovery.json").read_bytes()
        )
        assert initial.state == "running" and initial.sandbox_id == "sb-fixture"
        task.cancel("original interruption")
        await handle.stopping.wait()
        task.cancel("repeated interruption")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.5)
        assert handle.cleanup_finished.is_set()
        saved = remote.RecoveryReceipt.model_validate_json(
            (tmp_path / "report.recovery.json").read_bytes()
        )
        assert saved.state == "unknown" and saved.failure_type == "TimeoutError"
        assert saved.sandbox_id == "sb-fixture" and saved.volume_name == "volume"
        assert (
            saved.environment_name == "main" and saved.report_source == "learning/run-report.json"
        )
        assert "unconfirmed" in output.getvalue() and "sb-fixture" in output.getvalue()
        assert not handle.downloaded

    asyncio.run(exercise())


def test_owned_cleanup_timeout_cancels_and_joins_the_rpc() -> None:
    """A missing RPC response has a deadline even without another caller cancellation."""

    async def exercise() -> None:
        finished = asyncio.Event()

        async def stop() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        task = asyncio.create_task(stop())
        with pytest.raises(TimeoutError, match="foreground deadline"):
            await remote._join_cleanup(task, timeout_seconds=0.01)
        assert task.done() and task.cancelled() and finished.is_set()

    asyncio.run(exercise())

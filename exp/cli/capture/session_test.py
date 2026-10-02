"""Foreground failures and cancellation release interception before uploads."""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from exp.cli.capture import session
from exp.runtime.capture.control import CaptureRun, CaptureRunClient
from exp.runtime.capture.health import CaptureHealth, CaptureHealthFailure
from exp.runtime.capture.proxy import CaptureProxy
from exp.runtime.capture.upload import CaptureUploader, UploadStats


@pytest.fixture(autouse=True)
def healthy_network(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Keep session tests isolated from real provider DNS and network interception."""
    health = Mock(spec=CaptureHealth)
    health.check = AsyncMock(return_value=())

    async def watch() -> CaptureHealthFailure:
        """Represent a quiet healthy monitor until its session cancels it."""
        await asyncio.Event().wait()
        raise AssertionError("healthy monitor unexpectedly resumed")

    health.watch.side_effect = watch
    monkeypatch.setattr(session, "CaptureHealth", lambda domains, on_diagnostic=None: health)
    return health


@pytest.mark.parametrize("fail_before_ready", [False, True])
def test_proxy_failure_releases_interception_and_closes_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_before_ready: bool,
) -> None:
    """Inactive failures never announce capture; active failures drain uploads last."""
    events: list[str] = []
    proxy = Mock(spec=CaptureProxy)
    proxy.dropped_exchanges = 0
    uploader = Mock(spec=CaptureUploader)
    uploader.stats = UploadStats(0, 0, 0, 0, 0)
    uploader.start.side_effect = lambda: events.append("upload-start")
    uploader.close.side_effect = lambda **kwargs: events.append("upload-close")
    proxy.shutdown.side_effect = lambda: events.append("proxy-stop")

    async def serve(*, ca_directory: Path, ready: Callable[[], None]) -> None:
        """Fail either before approval or after reporting interception is active."""
        try:
            if fail_before_ready:
                raise RuntimeError("synthetic approval failure")
            ready()
            await asyncio.sleep(0.01)
            raise RuntimeError("synthetic active proxy failure")
        finally:
            events.append("interception-disabled")

    proxy.serve.side_effect = serve
    monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
    with pytest.raises(RuntimeError, match="synthetic"):
        asyncio.run(
            session.run_session(
                domains=("api.openai.com",),
                ca_directory=tmp_path,
                uploader=uploader,
                control=Mock(spec=CaptureRunClient),
                run=CaptureRun(
                    id=uuid4(),
                    org_id=uuid4(),
                    upload_origin="https://storage.example",
                    upload_path_prefix="/storage/v1/object/upload/sign/capture/",
                ),
                on_started=lambda: events.append("active"),
                on_progress=lambda stats: None,
                on_warning=lambda message: None,
            )
        )
    assert "interception-disabled" in events
    if fail_before_ready:
        assert "active" not in events
        assert "upload-start" not in events
        assert "upload-close" not in events
    else:
        assert "active" in events
        assert events[-1] == "upload-close"
        assert events.index("interception-disabled") < events.index("upload-close")


def test_cancellation_does_not_wait_for_macos_approval() -> None:
    """Ctrl+C during pending first-use approval returns promptly to cleanup."""

    async def run() -> None:
        """Keep startup pending while delivering the foreground stop signal."""
        stop = asyncio.Event()
        ready = asyncio.Event()

        async def pending() -> None:
            """Represent a backend waiting for OS approval."""
            await asyncio.Event().wait()

        task = asyncio.create_task(pending())
        stop.set()
        try:
            assert not await asyncio.wait_for(session._wait_for_proxy(task, ready, stop), 0.5)
            assert not ready.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("hang", [False, True])
def test_shutdown_failure_cannot_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hang: bool
) -> None:
    """A failed or stalled redirector stop stays a failure after uploader cleanup."""

    async def run() -> None:
        """Stop a simulated active session and inspect cleanup error propagation."""
        closing = asyncio.Event()
        proxy = Mock(spec=CaptureProxy)
        proxy.dropped_exchanges = 0
        proxy.shutdown.side_effect = closing.set
        uploader = Mock(spec=CaptureUploader)
        uploader.stats = UploadStats(0, 0, 0, 0, 0)
        original_wait = session._wait_for_proxy

        async def serve(*, ca_directory: Path, ready: Callable[[], None]) -> None:
            """Raise or remain pending when the session requests shutdown."""
            ready()
            await closing.wait()
            if hang:
                await asyncio.Event().wait()
            raise RuntimeError("synthetic redirector stop failure")

        async def wait(
            task: asyncio.Task[None],
            ready: asyncio.Event,
            stop: asyncio.Event,
            on_waiting: Callable[[], None] | None = None,
        ) -> bool:
            """Deliver the stop signal immediately after a successful startup."""
            result = await original_wait(task, ready, stop, on_waiting=on_waiting)
            asyncio.get_running_loop().call_soon(stop.set)
            return result

        proxy.serve.side_effect = serve
        monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
        monkeypatch.setattr(session, "_wait_for_proxy", wait)
        monkeypatch.setattr(session, "_SHUTDOWN_TIMEOUT", 0.02)
        with pytest.raises(RuntimeError, match="could not be confirmed|stop failure"):
            await session.run_session(
                domains=("api.openai.com",),
                ca_directory=tmp_path,
                uploader=uploader,
                control=Mock(spec=CaptureRunClient),
                run=CaptureRun(
                    id=uuid4(),
                    org_id=uuid4(),
                    upload_origin="https://storage.example",
                    upload_path_prefix="/storage/v1/object/upload/sign/capture/",
                ),
                on_started=lambda: None,
                on_progress=lambda stats: None,
                on_warning=lambda message: None,
            )
        uploader.close.assert_called_once()

    asyncio.run(run())


@pytest.mark.parametrize("finish", ["ready", "cancelled", "failed"])
@pytest.mark.parametrize("after_notice", [False, True])
def test_waiting_notice_respects_startup_completion(
    monkeypatch: pytest.MonkeyPatch,
    finish: Literal["ready", "cancelled", "failed"],
    after_notice: bool,
) -> None:
    """A pending startup reports once; a completed startup never reports waiting."""
    monkeypatch.setattr(session, "_WAITING_NOTICE_DELAY", 0.005)
    monkeypatch.setattr(session, "_STARTUP_TIMEOUT", 0.5)
    notices: list[str] = []

    async def run() -> None:
        """Complete the startup at the chosen point and retain orderly task cleanup."""
        ready = asyncio.Event()
        stop = asyncio.Event()
        noticed = asyncio.Event()

        def on_waiting() -> None:
            """Allow delayed startup to complete only once the notice was delivered."""
            notices.append("waiting")
            noticed.set()

        async def serve() -> None:
            """Represent startup completion without activating the network extension."""
            if after_notice:
                await noticed.wait()
                await asyncio.sleep(0.02)
            if finish == "failed":
                raise RuntimeError("synthetic startup failure")
            if finish == "ready":
                ready.set()
            else:
                stop.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(serve())
        try:
            if finish == "failed":
                with pytest.raises(RuntimeError, match="synthetic startup failure"):
                    await session._wait_for_proxy(task, ready, stop, on_waiting=on_waiting)
            else:
                assert await session._wait_for_proxy(task, ready, stop, on_waiting=on_waiting) is (
                    finish == "ready"
                )
            await asyncio.sleep(0.01)
            assert notices == (["waiting"] if after_notice else [])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("exhaust_deadline", [False, True])
def test_waiting_notice_does_not_extend_startup_deadline(
    monkeypatch: pytest.MonkeyPatch,
    exhaust_deadline: bool,
) -> None:
    """The second wait consumes only the original approval window's remaining time."""
    monkeypatch.setattr(session, "_WAITING_NOTICE_DELAY", 0.02)
    monkeypatch.setattr(session, "_STARTUP_TIMEOUT", 0.04)
    notices: list[str] = []
    timeouts: list[float] = []
    original_wait = asyncio.wait

    async def wait(
        tasks: set[asyncio.Task[None] | asyncio.Task[bool]],
        *,
        timeout: float,
        return_when: str,
    ) -> tuple[
        set[asyncio.Task[None] | asyncio.Task[bool]],
        set[asyncio.Task[None] | asyncio.Task[bool]],
    ]:
        """Record real event-loop wait budgets without bypassing their timeouts."""
        timeouts.append(timeout)
        return await original_wait(tasks, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(session.asyncio, "wait", wait)

    def on_waiting() -> None:
        """Exercise both a prompt notice and a notice that exhausts the startup budget."""
        notices.append("waiting")
        if exhaust_deadline:
            time.sleep(session._STARTUP_TIMEOUT)

    async def run() -> None:
        """Leave startup pending through the notice and the finite approval deadline."""
        task = asyncio.create_task(asyncio.sleep(1.0))
        try:
            with pytest.raises(RuntimeError, match="Login Items & Extensions > Network Extensions"):
                await session._wait_for_proxy(
                    task,
                    asyncio.Event(),
                    asyncio.Event(),
                    on_waiting=on_waiting,
                )
            assert notices == ["waiting"]
            assert len(timeouts) == 2
            assert 0 <= timeouts[1] <= max(0, timeouts[0] - session._WAITING_NOTICE_DELAY)
            if exhaust_deadline:
                assert timeouts[1] == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


async def _run_network_session(
    directory: Path,
    uploader: Mock,
    events: list[str],
) -> UploadStats:
    """Drive foreground health handling with only synthetic network and upload owners."""
    return await session.run_session(
        domains=("api.openai.com",),
        ca_directory=directory,
        uploader=uploader,
        control=Mock(spec=CaptureRunClient),
        run=CaptureRun(
            id=uuid4(),
            org_id=uuid4(),
            upload_origin="https://storage.example",
            upload_path_prefix="/storage/v1/object/upload/sign/capture/",
        ),
        on_started=lambda: events.append("active"),
        on_progress=lambda stats: None,
        on_warning=lambda message: events.append(message),
    )


@pytest.mark.parametrize("kind", ["dns", "monitor"])
def test_failed_network_baseline_never_starts_interception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    healthy_network: Mock,
    kind: Literal["dns", "monitor"],
) -> None:
    """A broken connection or unavailable check cannot start the native redirector."""
    proxy = Mock(spec=CaptureProxy)
    uploader = Mock(spec=CaptureUploader)
    events: list[str] = []
    healthy_network.check.return_value = (CaptureHealthFailure("api.openai.com", kind),)
    monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
    with pytest.raises(RuntimeError, match="Cannot resolve|could not check network"):
        asyncio.run(_run_network_session(tmp_path, uploader, events))
    proxy.serve.assert_not_called()
    uploader.start.assert_not_called()
    healthy_network.watch.assert_not_called()
    assert events == []


def test_initial_network_check_is_cancelled_with_startup(healthy_network: Mock) -> None:
    """Interrupting a slow baseline cancels its owned probe instead of delaying Ctrl+C."""

    async def run() -> None:
        """Deliver the stop signal only once a baseline probe has started."""
        stop = asyncio.Event()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def check() -> tuple[CaptureHealthFailure, ...]:
            """Keep one probe pending until the startup helper cancels it."""
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return ()

        healthy_network.check.side_effect = check
        task = asyncio.create_task(session._check_before_capture(healthy_network, stop))
        await started.wait()
        stop.set()
        assert await asyncio.wait_for(task, 0.5) is None
        assert cancelled.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("recovered", [True, False])
@pytest.mark.parametrize("unrelated_failure", [False, True])
def test_network_guard_stops_interception_before_checking_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    healthy_network: Mock,
    recovered: bool,
    unrelated_failure: bool,
) -> None:
    """Persistent failure stops forwarding, then reports the actual post-stop DNS result."""

    async def run() -> None:
        """Trigger the monitor while a synthetic native owner remains active."""
        events: list[str] = []
        closing = asyncio.Event()
        proxy = Mock(spec=CaptureProxy)
        proxy.dropped_exchanges = 0
        proxy.shutdown.side_effect = closing.set
        uploader = Mock(spec=CaptureUploader)
        uploader.stats = UploadStats(0, 0, 0, 0, 0)
        uploader.close.side_effect = lambda **kwargs: events.append("upload-close")
        failure = CaptureHealthFailure("api.openai.com", "dns")

        async def serve(*, ca_directory: Path, ready: Callable[[], None]) -> None:
            """Keep interception alive until the guard requests orderly shutdown."""
            ready()
            await closing.wait()
            events.append("interception-disabled")

        async def check() -> tuple[CaptureHealthFailure, ...]:
            """Distinguish startup baseline from post-shutdown verification."""
            if not events:
                return ()
            assert "interception-disabled" in events
            events.append("recovery-checked")
            unrelated = (
                (CaptureHealthFailure("api.anthropic.com", "dns"),) if unrelated_failure else ()
            )
            return unrelated + (() if recovered else (failure,))

        async def watch() -> CaptureHealthFailure:
            """Signal a sustained DNS regression after active capture was announced."""
            assert events == ["active"]
            return failure

        proxy.serve.side_effect = serve
        healthy_network.check.side_effect = check
        healthy_network.watch.side_effect = watch
        monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
        with pytest.raises(RuntimeError) as error:
            await _run_network_session(tmp_path, uploader, events)
        expected_error = (
            "Capture stopped after repeated DNS failures for api.openai.com. "
            "DNS is responding again."
            if recovered
            else "Capture stopped, but DNS for api.openai.com is still unavailable. "
            "Check your connection before retrying."
        )
        assert str(error.value) == expected_error
        expected = ["active", "interception-disabled", "recovery-checked", "upload-close"]
        if unrelated_failure:
            expected.append(
                "DNS for api.anthropic.com is unavailable after Capture stopped; "
                "check your connection before retrying."
            )
        assert events == expected
        assert proxy.serve.await_count == 1

    asyncio.run(run())


@pytest.mark.parametrize("final_dns_failure", [False, True])
def test_normal_stop_checks_dns_and_cancels_network_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    healthy_network: Mock,
    final_dns_failure: bool,
) -> None:
    """Ctrl+C stays quiet when healthy and describes a new final failure without history."""

    async def run() -> None:
        """Stop after native readiness while the monitor is inside a bounded DNS check."""
        events: list[str] = []
        closing = asyncio.Event()
        watching = asyncio.Event()
        cancelled = asyncio.Event()
        proxy = Mock(spec=CaptureProxy)
        proxy.dropped_exchanges = 0
        proxy.shutdown.side_effect = closing.set
        uploader = Mock(spec=CaptureUploader)
        uploader.stats = UploadStats(0, 0, 0, 0, 0)
        original_wait = session._wait_for_proxy

        async def serve(*, ca_directory: Path, ready: Callable[[], None]) -> None:
            """Remain active until normal session teardown requests a stop."""
            ready()
            await closing.wait()

        async def watch() -> CaptureHealthFailure:
            """Record that teardown cancels and waits for the active health check."""
            watching.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError("monitor unexpectedly resumed")

        async def wait(
            task: asyncio.Task[None],
            ready: asyncio.Event,
            stop: asyncio.Event,
            on_waiting: Callable[[], None] | None = None,
        ) -> bool:
            """Schedule interruption only after the session starts its monitor."""
            result = await original_wait(task, ready, stop, on_waiting)

            async def interrupt() -> None:
                """Simulate Ctrl+C once the health probe is pending."""
                await watching.wait()
                stop.set()

            asyncio.create_task(interrupt())
            return result

        proxy.serve.side_effect = serve
        healthy_network.watch.side_effect = watch
        healthy_network.check.side_effect = [
            (),
            (CaptureHealthFailure("api.openai.com", "dns"),) if final_dns_failure else (),
        ]
        monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
        monkeypatch.setattr(session, "_wait_for_proxy", wait)
        if final_dns_failure:
            with pytest.raises(RuntimeError, match="DNS for api.openai.com is unavailable"):
                await _run_network_session(tmp_path, uploader, events)
        else:
            assert await _run_network_session(tmp_path, uploader, events) == uploader.stats
        assert cancelled.is_set()
        assert healthy_network.check.await_count == 2
        assert events == ["active"]

    asyncio.run(run())


@pytest.mark.parametrize("probe_raises", [False, True])
def test_backend_failure_checks_dns_without_masking_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    healthy_network: Mock,
    probe_raises: bool,
) -> None:
    """Report lingering DNS trouble while preserving the original backend failure."""
    proxy = Mock(spec=CaptureProxy)
    proxy.dropped_exchanges = 0
    uploader = Mock(spec=CaptureUploader)
    uploader.stats = UploadStats(0, 0, 0, 0, 0)
    events: list[str] = []

    async def serve(*, ca_directory: Path, ready: Callable[[], None]) -> None:
        """Fail after readiness so session cleanup can inspect the affected network path."""
        ready()
        await asyncio.sleep(0.01)
        raise RuntimeError("synthetic native backend failure")

    healthy_network.check.side_effect = [
        (),
        RuntimeError("synthetic health check failure")
        if probe_raises
        else (CaptureHealthFailure("api.openai.com", "dns"),),
    ]
    proxy.serve.side_effect = serve
    monkeypatch.setattr(session, "CaptureProxy", lambda **kwargs: proxy)
    with pytest.raises(RuntimeError, match="synthetic native backend failure"):
        asyncio.run(_run_network_session(tmp_path, uploader, events))
    assert healthy_network.check.await_count == 2
    assert events == (
        ["active"]
        if probe_raises
        else [
            "active",
            "DNS for api.openai.com is also unavailable; "
            "check your connection before retrying Capture.",
        ]
    )
    uploader.close.assert_called_once()

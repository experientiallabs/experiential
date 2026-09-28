"""Volume acknowledgement failures stay visible without contacting Modal."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from exp.optimize.claas.backends.modal import persistence


def test_commit_uses_mount_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commit all files and metadata on the mounted v2 Volume before returning."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        sync = AsyncMock()
        monkeypatch.setattr(persistence, "sync_mount", sync)
        await persistence.commit_volume()
        sync.assert_awaited_once_with(Path("/state"))

    asyncio.run(scenario())


def test_failed_sync_is_not_acknowledged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonzero filesystem commit is a persistence failure, not a successful flush."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        process = AsyncMock()
        process.wait.return_value = 1
        spawn = AsyncMock(return_value=process)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        with pytest.raises(RuntimeError, match="Volume commit failed"):
            await persistence.sync_mount(Path("/state"))
        assert spawn.call_args.args == ("/usr/bin/sync", "/state")

    asyncio.run(scenario())


def test_cancelled_sync_reaps_owned_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation is preserved after killing and joining the filesystem commit process."""

    async def scenario() -> None:
        """Cancel a pending commit and force a second cancellation while it is being reaped."""
        running = asyncio.Event()
        reaping = asyncio.Event()
        finished = asyncio.Event()
        calls = 0

        async def wait() -> int:
            """Hold the initial process and then hold reaping until explicitly released."""
            nonlocal calls
            calls += 1
            if calls == 1:
                running.set()
                await asyncio.Future()
            reaping.set()
            await finished.wait()
            return -9

        process = AsyncMock()
        process.returncode = None
        process.wait.side_effect = wait
        process.kill = Mock()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
        task = asyncio.create_task(persistence.sync_mount(Path("/state")))
        await running.wait()
        task.cancel("initial")
        await reaping.wait()
        task.cancel("again")
        await asyncio.sleep(0)
        assert not task.done()
        finished.set()
        with pytest.raises(asyncio.CancelledError, match="initial"):
            await task
        process.kill.assert_called_once()
        assert calls == 2

    asyncio.run(scenario())


def test_mount_sync_cannot_be_substituted_through_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writable PATH cannot replace the Linux system commit executable or its absence."""
    marker = tmp_path / "substituted"
    substitute = tmp_path / "sync"
    substitute.write_text(f"#!/bin/sh\n: > '{marker}'\n")
    substitute.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    try:
        asyncio.run(persistence.sync_mount(tmp_path))
    except FileNotFoundError as error:
        # Non-Linux developer hosts may lack Modal's required system executable.
        assert error.filename == "/usr/bin/sync"
    assert not marker.exists()

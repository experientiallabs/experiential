"""Approval navigation requires an exact pending extension and never mutates it."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from exp.cli.capture import approval

_PENDING = (
    "1 extension(s)\n"
    "enabled\tactive\tteamID\tbundleID (version)\tname\t[state]\n"
    "\t*\tS8XHQB96PW\torg.mitmproxy.macos-redirector.network-extension (2.0/1)\t"
    "network-extension\t[activated waiting for user]\n"
)


def _process(output: str = "", returncode: int | None = 0) -> Mock:
    """Create a finite synthetic subprocess without invoking any system command."""
    process = Mock(spec=asyncio.subprocess.Process)
    process.returncode = returncode
    process.communicate = AsyncMock(return_value=(output.encode(), b""))
    process.wait = AsyncMock(return_value=0)
    return process


def test_pending_approval_opens_settings_without_toggling_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmed pending extension navigates to the verified Settings route only."""
    monkeypatch.setattr(approval.sys, "platform", "darwin")
    create = AsyncMock(side_effect=[_process(_PENDING), _process()])
    monkeypatch.setattr(approval.asyncio, "create_subprocess_exec", create)
    assert asyncio.run(approval.open_pending_approval_settings())
    assert [call.args for call in create.call_args_list] == [
        ("/usr/bin/systemextensionsctl", "list"),
        ("/usr/bin/open", "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"),
    ]
    for call in create.call_args_list:
        assert call.kwargs == {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.DEVNULL,
        }


@pytest.mark.parametrize(
    "status",
    [
        "0 extension(s)",
        "unrecognized output",
        _PENDING.replace("activated waiting for user", "activated enabled"),
        _PENDING.replace("activated waiting for user", "activated enabling"),
        _PENDING.replace("network-extension (", "network-extension-other ("),
        _PENDING.replace("S8XHQB96PW", "OTHERTEAM1"),
        "status says org.mitmproxy.macos-redirector.network-extension [activated waiting for user]",
    ],
)
def test_unconfirmed_approval_never_opens_settings(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Ready, unknown, and unrelated extension states cannot trigger navigation."""
    monkeypatch.setattr(approval.sys, "platform", "darwin")
    create = AsyncMock(return_value=_process(status))
    monkeypatch.setattr(approval.asyncio, "create_subprocess_exec", create)
    assert not asyncio.run(approval.open_pending_approval_settings())
    assert create.call_count == 1


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_do_not_inspect_extensions(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    """The optional macOS setup helper remains inert on other systems."""
    monkeypatch.setattr(approval.sys, "platform", platform)
    create = AsyncMock(side_effect=AssertionError("unexpected subprocess"))
    monkeypatch.setattr(approval.asyncio, "create_subprocess_exec", create)
    assert not asyncio.run(approval.open_pending_approval_settings())
    create.assert_not_called()


@pytest.mark.parametrize("failure", ["nonzero", "missing", "encoding"])
@pytest.mark.parametrize("stage", ["inspection", "navigation"])
def test_optional_command_failure_does_not_fail_capture(
    monkeypatch: pytest.MonkeyPatch, failure: str, stage: str
) -> None:
    """Unavailable status or navigation remains best effort with no thrown setup error."""
    monkeypatch.setattr(approval.sys, "platform", "darwin")
    result: Mock | Exception
    if failure == "nonzero":
        result = _process(returncode=1)
    elif failure == "missing":
        result = FileNotFoundError("synthetic")
    else:
        result = _process()
        result.communicate.return_value = (b"\xff", b"")
    effects = [_process(_PENDING), result] if stage == "navigation" else [result]
    create = AsyncMock(side_effect=effects)
    monkeypatch.setattr(approval.asyncio, "create_subprocess_exec", create)
    assert not asyncio.run(approval.open_pending_approval_settings())
    assert create.call_count == (2 if stage == "navigation" else 1)


@pytest.mark.parametrize("cancelled", [False, True])
def test_pending_inspection_is_bounded_and_cancellable(
    monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    """An unfinished status process is killed and reaped without opening Settings."""
    monkeypatch.setattr(approval.sys, "platform", "darwin")
    monkeypatch.setattr(approval, "_COMMAND_TIMEOUT", 0.02)
    process = _process(returncode=None)
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(approval.asyncio, "create_subprocess_exec", create)

    async def run() -> None:
        """Timeout or cancel a synthetic read-only process during its status query."""
        inspected = asyncio.Event()

        async def communicate() -> tuple[bytes, bytes]:
            """Remain pending while the event loop handles timeout or cancellation."""
            inspected.set()
            await asyncio.Event().wait()
            return b"", b""

        process.communicate.side_effect = communicate
        task = asyncio.create_task(approval.open_pending_approval_settings())
        await inspected.wait()
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.2)
        else:
            assert not await asyncio.wait_for(task, timeout=0.2)

    asyncio.run(run())
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()
    assert create.call_count == 1

"""Exercise independent cleanup against real processes and a fake extension control socket."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from exp.runtime.capture.watchdog import CaptureWatchdog, _frame

_OWNER = """
import asyncio, os, socket, sys, time
from exp.runtime.capture import watchdog
watchdog._HEARTBEAT_TIMEOUT = 0.5
async def main():
    control = socket.socket(fileno=int(sys.argv[1]))
    guard = await watchdog.CaptureWatchdog.start(control)
    await guard.activate()
    os.write(1, (str(guard._process.pid) + "\\n").encode())
    if sys.argv[2] == "hang":
        time.sleep(30)
    elif sys.argv[2] == "clean":
        await guard.close()
    else:
        try:
            await guard.wait_failed()
        finally:
            await guard.close()
asyncio.run(main())
"""


def _receive(control: socket.socket) -> bytes:
    """Read one exact length-prefixed protobuf frame from the simulated extension."""
    header = bytearray()
    while len(header) < 4:
        chunk = control.recv(4 - len(header))
        assert chunk, "control closed before frame header"
        header.extend(chunk)
    length = struct.unpack("!I", header)[0]
    body = bytearray()
    while len(body) < length:
        chunk = control.recv(length - len(body))
        assert chunk, "control closed before frame body"
        body.extend(chunk)
    return bytes(header + body)


@pytest.mark.parametrize("failure", ["crash", "hang", "clean", "watchdog"])
def test_watchdog_disables_after_owner_or_watchdog_failure(failure: str) -> None:
    """Owner death, a blocked loop, and watchdog death all disable the same owned connection."""
    extension, control = socket.socketpair()
    extension.settimeout(7)
    owner = subprocess.Popen(
        [sys.executable, "-c", _OWNER, str(control.fileno()), failure],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(control.fileno(),),
        start_new_session=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    control.close()
    try:
        assert owner.stdout is not None
        watchdog_pid = int(owner.stdout.readline())
        assert _receive(extension) == _frame(
            ("!mDNSResponder", f"!{owner.pid}", f"!{watchdog_pid}")
        )
        if failure == "crash":
            owner.kill()
        elif failure == "watchdog":
            os.kill(watchdog_pid, signal.SIGKILL)
        assert _receive(extension) == _frame(("0", "!0"))
        owner.wait(timeout=7)
        if failure == "clean":
            assert owner.returncode == 0
        else:
            assert owner.returncode != 0
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=5)
        extension.close()


def test_responsive_watchdog_keeps_interception_and_normal_close_releases_descriptors() -> None:
    """Heartbeats preserve an active session until acknowledged shutdown closes the socket."""

    async def scenario() -> None:
        """Run the production lifecycle with a socket pair and no network extension."""
        extension, control = socket.socketpair()
        extension.setblocking(False)
        guard = await CaptureWatchdog.start(control)
        try:
            await guard.activate()
            data = await asyncio.get_running_loop().sock_recv(extension, 4096)
            assert data == _frame(("!mDNSResponder", f"!{os.getpid()}", f"!{guard._process.pid}"))
            await asyncio.sleep(1.1)
            assert not guard._heartbeat.done()
            await guard.close()
            assert guard._process.returncode == 0
            assert control.fileno() == -1
            remaining = bytearray()
            while chunk := await asyncio.get_running_loop().sock_recv(extension, 4096):
                remaining.extend(chunk)
            assert remaining == _frame(("0", "!0")) * 3
        finally:
            extension.close()

    asyncio.run(scenario())

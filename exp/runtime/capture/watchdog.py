"""Disable an owned macOS redirector independently of Capture's event loop."""

from __future__ import annotations

import asyncio
import logging
import os
import select
import signal
import socket
import struct
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
_HEARTBEAT_TIMEOUT = 15.0
_DISABLED_ACTIONS = ("0", "!0")


def _frame(actions: tuple[str, ...]) -> bytes:
    """Encode bounded InterceptConf strings using the redirector's existing protobuf wire format."""
    body = bytearray()
    for action in actions:
        encoded = action.encode("utf-8")
        if not 0 < len(encoded) < 128:
            raise ValueError("Invalid capture interception action.")
        body.extend((10, len(encoded)))
        body.extend(encoded)
    return struct.pack("!I", len(body)) + body


def _write_control(descriptor: int, actions: tuple[str, ...]) -> None:
    """Write one complete configuration without changing the shared socket's blocking mode."""
    remaining = memoryview(_frame(actions))
    deadline = time.monotonic() + 2.0
    while remaining:
        wait = deadline - time.monotonic()
        if wait <= 0 or not select.select([], [descriptor], [], wait)[1]:
            raise TimeoutError("Capture could not disable interception.")
        try:
            sent = os.write(descriptor, remaining)
        except BlockingIOError:
            continue
        if sent == 0:
            raise OSError("Capture control socket closed.")
        remaining = remaining[sent:]


def _supervise(descriptor: int, owner: int, timeout: float) -> None:
    """Own all control writes and retire only this process's parent if its heartbeat expires."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.write(1, b"ready\n")
    deadline = time.monotonic() + timeout
    clean = False
    try:
        while True:
            wait = max(0.0, deadline - time.monotonic())
            if not select.select([0], [], [], wait)[0]:
                break
            commands = os.read(0, 4096)
            if not commands:
                break
            for command in commands:
                if command == ord("A"):
                    _write_control(descriptor, ("!mDNSResponder", f"!{owner}", f"!{os.getpid()}"))
                    os.write(1, b"active\n")
                elif command == ord("D"):
                    _write_control(descriptor, _DISABLED_ACTIONS)
                    os.write(1, b"disabled\n")
                elif command == ord("X"):
                    clean = True
                    return
                elif command != ord("H"):
                    raise ValueError("Invalid Capture watchdog command.")
            deadline = time.monotonic() + timeout
    finally:
        try:
            _write_control(descriptor, _DISABLED_ACTIONS)
        finally:
            os.close(descriptor)
            if not clean and os.getppid() == owner:
                logger.error("Capture stopped responding; stopping Capture to restore connections.")
                # The child-parent relationship identifies the exact owner, even if a PID
                # could otherwise be recycled. Never signal another process after reparenting.
                os.kill(owner, signal.SIGTERM)
                deadline = time.monotonic() + 2.0
                while os.getppid() == owner and time.monotonic() < deadline:
                    time.sleep(0.05)
                if os.getppid() == owner:
                    os.kill(owner, signal.SIGKILL)


class CaptureWatchdog:
    """Keep the independent control owner alive while the foreground loop is responsive."""

    def __init__(self, process: asyncio.subprocess.Process, control: socket.socket) -> None:
        """Retain a recovery descriptor in case the watchdog itself exits unexpectedly."""
        self._process = process
        self._control = control
        self._heartbeat = asyncio.create_task(self._pulse())

    @classmethod
    async def start(cls, control: socket.socket) -> CaptureWatchdog:
        """Start and acknowledge the watchdog before any application is intercepted."""
        process: asyncio.subprocess.Process | None = None
        control.setblocking(False)
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).resolve()),
                str(control.fileno()),
                str(os.getpid()),
                str(_HEARTBEAT_TIMEOUT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,
                pass_fds=(control.fileno(),),
                start_new_session=True,
            )
            assert process.stdout is not None
            if await asyncio.wait_for(process.stdout.readline(), timeout=5.0) != b"ready\n":
                raise RuntimeError(
                    "Capture could not start its network watchdog. Retry exp capture."
                )
            return cls(process, control)
        except BaseException:
            if process is not None:
                if process.returncode is None:
                    process.kill()
                await process.wait()
            control.close()
            raise

    async def _pulse(self) -> None:
        """Send one byte per second; foreground stalls cannot renew this deadline."""
        assert self._process.stdin is not None
        while self._process.returncode is None:
            self._process.stdin.write(b"H")
            await self._process.stdin.drain()
            await asyncio.sleep(1)
        raise RuntimeError("Capture's network watchdog stopped. Retry exp capture.")

    async def activate(self) -> None:
        """Enable provider interception only after watchdog ownership is established."""
        await self._command(b"A", b"active\n")

    async def disable(self) -> None:
        """Acknowledge disabling interception, or recover directly after a watchdog exit."""
        if self._process.returncode is None:
            await self._command(b"D", b"disabled\n")
        else:
            await asyncio.wait_for(
                asyncio.get_running_loop().sock_sendall(self._control, _frame(_DISABLED_ACTIONS)),
                timeout=2.0,
            )

    async def _command(self, command: bytes, expected: bytes) -> None:
        """Serialize lifecycle commands with their watchdog acknowledgements."""
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(command)
        await self._process.stdin.drain()
        if await asyncio.wait_for(self._process.stdout.readline(), timeout=3.0) != expected:
            raise RuntimeError("Capture lost its network watchdog. Retry exp capture.")

    async def wait_failed(self) -> None:
        """Expose the heartbeat task without giving observers ownership of its lifetime."""
        await asyncio.shield(self._heartbeat)

    async def close(self) -> None:
        """Finish after interception and native forwarding have stopped."""
        try:
            await self.disable()
            if self._process.returncode is None:
                assert self._process.stdin is not None
                self._process.stdin.write(b"X")
                await self._process.stdin.drain()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
        finally:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
            if self._process.returncode is None:
                self._process.kill()
                await self._process.wait()
            try:
                await asyncio.wait_for(
                    asyncio.get_running_loop().sock_sendall(
                        self._control, _frame(_DISABLED_ACTIONS)
                    ),
                    timeout=2.0,
                )
            except (BrokenPipeError, ConnectionResetError):
                # The extension has already closed its end of the owned channel.
                pass
            finally:
                self._control.close()


if __name__ == "__main__":
    _supervise(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))

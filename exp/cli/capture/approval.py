"""Open macOS approval settings only after confirming Capture's extension is pending."""

from __future__ import annotations

import asyncio
import re
import sys

_COMMAND_TIMEOUT = 2.0
_SETTINGS_URL = "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"
_PENDING_EXTENSION = re.compile(
    r"^\s*(?:\*\s+){0,2}S8XHQB96PW\s+"
    r"org\.mitmproxy\.macos-redirector\.network-extension\s+"
    r"\([^()\s]+\)\s+\S+\s+\[activated waiting for user\]\s*$"
)


async def open_pending_approval_settings() -> bool:
    """Navigate to Settings when macOS reports this extension awaiting user approval.

    Inspection and navigation use asynchronous processes with finite command timeouts.
    Cancellation while inspecting prevents subsequent navigation. The user owns
    the approval action; this function never enables or disables an extension.

    Returns:
        True if the pending state was confirmed and macOS accepted the Settings URL.
        False on other platforms, unrecognized status, or an unavailable command.
    """
    if sys.platform != "darwin":
        return False
    status = await _run(["/usr/bin/systemextensionsctl", "list"])
    if status is None or not any(
        _PENDING_EXTENSION.fullmatch(line) for line in status.splitlines()
    ):
        return False
    opened = await _run(["/usr/bin/open", _SETTINGS_URL])
    return opened is not None


async def _run(command: list[str]) -> str | None:
    """Bound each optional child process and reap it when cancelled or timed out."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
    except OSError:
        return None
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=_COMMAND_TIMEOUT)
    except (OSError, TimeoutError):
        return None
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=_COMMAND_TIMEOUT)
            except (OSError, TimeoutError):
                pass
    if process.returncode != 0:
        return None
    try:
        return stdout.decode("utf-8")
    except UnicodeError:
        return None

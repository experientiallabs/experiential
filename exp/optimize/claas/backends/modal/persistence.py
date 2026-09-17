"""Explicit v2 Volume commits for the generic controller's persistence callback."""

import asyncio
from pathlib import Path


async def commit_volume() -> None:
    """Persist mounted state before acknowledgement; a nonzero sync fails closed.

    Modal documents ``sync /state`` as an explicit v2 Volume commit. Its Linux image
    must supply the trusted system executable at ``/usr/bin/sync``. The hosting
    adapter validates the Volume version before launch. SQLite must finish its
    DELETE-journal transaction before invoking this callback. Never reload a
    mounted Volume while the controller owns its database.
    """
    await sync_mount(Path("/state"))


async def sync_mount(mount: Path, *, timeout_seconds: float = 120) -> None:
    """Run a finite filesystem commit and always reap the owned sync subprocess."""
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/sync",
        str(mount),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            code = await process.wait()
    except BaseException:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        cleanup = asyncio.create_task(process.wait())
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
        raise
    if code != 0:
        raise RuntimeError(
            "Modal Volume commit failed; retry persistence before acknowledging work"
        )

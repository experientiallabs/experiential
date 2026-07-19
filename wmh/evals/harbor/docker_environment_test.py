"""Tests for the cancellation-safe Harbor Docker environment boundary."""

from __future__ import annotations

import asyncio
import sys

import pytest

from wmh.evals.harbor.docker_environment import ReapingDockerEnvironment


def test_buffered_command_cancellation_kills_and_reaps_process() -> None:
    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        collection = asyncio.create_task(
            ReapingDockerEnvironment._collect_buffered_output(
                process,
                timeout_sec=None,
            )
        )
        await asyncio.sleep(0.05)

        collection.cancel()
        with pytest.raises(asyncio.CancelledError):
            await collection

        assert process.returncode is not None
        assert await process.wait() == process.returncode

    asyncio.run(scenario())


def test_buffered_command_timeout_kills_and_reaps_process() -> None:
    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        with pytest.raises(RuntimeError, match="timed out after 0.05 seconds"):
            await ReapingDockerEnvironment._collect_buffered_output(
                process,
                timeout_sec=0.05,
            )

        assert process.returncode is not None
        assert await process.wait() == process.returncode

    asyncio.run(scenario())


def test_buffered_command_preserves_completed_output() -> None:
    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('ready'); sys.stderr.write('note')",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        result = await ReapingDockerEnvironment._collect_buffered_output(
            process,
            timeout_sec=1,
        )

        assert result.return_code == 0
        assert result.stdout == "ready"
        assert result.stderr == "note"

    asyncio.run(scenario())


def test_zero_timeout_preserves_harbor_no_timeout_semantics() -> None:
    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('ready')",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        result = await ReapingDockerEnvironment._collect_buffered_output(
            process,
            timeout_sec=0,
        )

        assert result.return_code == 0
        assert result.stdout == "ready"

    asyncio.run(scenario())

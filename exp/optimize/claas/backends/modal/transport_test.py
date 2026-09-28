"""Transfer boundaries preserve state and avoid unbounded artifact downloads."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import modal
import pytest

from exp.optimize.claas.backends.modal.transport import (
    download_artifact,
    upload_import,
)


def test_import_does_not_overwrite_existing_run(tmp_path: Path) -> None:
    """A retried launch cannot change an already staged exact-evidence artifact."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        filesystem = SimpleNamespace(
            make_directory=SimpleNamespace(aio=AsyncMock()),
            list_files=SimpleNamespace(
                aio=AsyncMock(return_value=[SimpleNamespace(name="run-1.jsonl")])
            ),
            write_bytes=SimpleNamespace(aio=AsyncMock()),
        )
        sandbox = cast(modal.Sandbox, SimpleNamespace(filesystem=filesystem))
        path = tmp_path / "input.jsonl"
        path.write_text("{}")
        with pytest.raises(ValueError, match="new run_id"):
            await upload_import(sandbox, path, "run-1", maximum_bytes=10)
        filesystem.write_bytes.aio.assert_not_called()

    asyncio.run(scenario())


def test_oversized_download_removes_partial_file(tmp_path: Path) -> None:
    """A rejected download leaves no plausible truncated checkpoint behind."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""

        async def chunks(path: str) -> AsyncIterator[bytes]:
            """Serve a deterministic committed artifact without a Modal account."""
            assert path == "report.json"
            yield b"123"
            yield b"456"

        volume = cast(modal.Volume, SimpleNamespace(read_file=SimpleNamespace(aio=chunks)))
        destination = tmp_path / "report.json"
        with pytest.raises(ValueError, match="maximum_bytes"):
            await download_artifact(volume, "report.json", destination, maximum_bytes=4)
        assert not destination.exists()

    asyncio.run(scenario())


def test_download_preserves_existing_file(tmp_path: Path) -> None:
    """An existing local artifact cannot be replaced implicitly by a remote download."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        destination = tmp_path / "report.json"
        destination.write_text("existing")
        with pytest.raises(FileExistsError):
            await download_artifact(
                cast(modal.Volume, object()), "report.json", destination, maximum_bytes=4
            )
        assert destination.read_text() == "existing"

    asyncio.run(scenario())

"""Cross-App ownership and persisted bindings without a live Modal account."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import modal
import pytest

from exp.optimize.claas.backends.modal.ownership import bind_owner, commit_owner


def volume(marker: bytes | None) -> modal.Volume:
    """Expose the committed owner marker through the published streaming interface."""

    async def read(path: str) -> AsyncIterator[bytes]:
        """Read only the reserved, bounded ownership artifact."""
        assert path == ".claas-modal-owner"
        if marker is None:
            raise FileNotFoundError(path)
        yield marker

    return cast(
        modal.Volume,
        SimpleNamespace(
            object_id="vo-state",
            read_file=SimpleNamespace(aio=read),
        ),
    )


def test_first_use_atomically_binds_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-App launchers use the same Volume identity and conditional create."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        registry = SimpleNamespace(
            put=SimpleNamespace(aio=AsyncMock(return_value=True)),
            get=SimpleNamespace(aio=AsyncMock(return_value="learning")),
        )
        factory = Mock(return_value=registry)
        monkeypatch.setattr(modal.Dict, "from_name", factory)
        await bind_owner(volume(None), app_name="learning", environment_name="main")
        registry.put.aio.assert_awaited_once_with("vo-state", "learning", skip_if_exists=True)
        assert factory.call_args.args == ("experiential-claas-volume-owners",)

    asyncio.run(scenario())


def test_competing_app_rejected_before_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """The conditional winner retains sole authority to create the named Sandbox."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        registry = SimpleNamespace(
            put=SimpleNamespace(aio=AsyncMock(return_value=False)),
            get=SimpleNamespace(aio=AsyncMock(return_value="first-app")),
        )
        monkeypatch.setattr(modal.Dict, "from_name", Mock(return_value=registry))
        with pytest.raises(ValueError, match="original App"):
            await bind_owner(volume(None), app_name="second-app", environment_name="main")

    asyncio.run(scenario())


def test_idle_dict_expiry_cannot_reassign_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed marker keeps ownership when the seven-day Dict entry expires."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        factory = Mock()
        monkeypatch.setattr(modal.Dict, "from_name", factory)
        with pytest.raises(ValueError, match="original App"):
            await bind_owner(volume(b"first-app"), app_name="second-app", environment_name="main")
        factory.assert_not_called()

    asyncio.run(scenario())


def test_failed_owner_commit_stops_before_service() -> None:
    """A failed v2 commit cannot authorize starting the learner with transient ownership."""

    async def scenario() -> None:
        """Exercise the asynchronous hosting contract in one owned event loop."""
        command = SimpleNamespace(wait=SimpleNamespace(aio=AsyncMock()), returncode=1)
        sandbox = cast(
            modal.Sandbox,
            SimpleNamespace(
                filesystem=SimpleNamespace(write_text=SimpleNamespace(aio=AsyncMock())),
                exec=SimpleNamespace(aio=AsyncMock(return_value=command)),
            ),
        )
        with pytest.raises(RuntimeError, match="ownership commit failed"):
            await commit_owner(sandbox, "learning")

    asyncio.run(scenario())

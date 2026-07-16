# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for persisted detached-session references and workspace checkpoints."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from wmh.cli.session_state import (
    DetachedSessionState,
    SessionStateError,
    SessionStateStore,
    WorkspaceCheckpoint,
)


def _state(
    session_id: str = "sess-1", *, workspace: WorkspaceCheckpoint | None = None
) -> DetachedSessionState:
    """A representative detached-session reference."""
    return DetachedSessionState(
        api_url="https://api.test",
        web_url="https://platform.test",
        agent_id="agent-1",
        agent_name="Support Agent",
        session_id=session_id,
        created_at="2026-07-15T00:00:00+00:00",
        workspace=workspace,
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """A saved reference loads back typed and equal."""
    store = SessionStateStore(tmp_path)
    saved = store.save(_state())

    loaded = store.load("sess-1")

    assert loaded == saved
    assert loaded is not None
    assert loaded.agent_id == "agent-1"
    assert loaded.cursor == 0
    assert loaded.workspace is None


def test_load_missing_session_returns_none(tmp_path: Path) -> None:
    """An unknown session id is a clean miss, not an error."""
    assert SessionStateStore(tmp_path).load("sess-unknown") is None


def test_state_files_are_owner_only(tmp_path: Path) -> None:
    """The state directory and its files never leak to other users."""
    directory = tmp_path / "sessions"
    store = SessionStateStore(directory)
    store.save(_state())
    store.set_current("sess-1")

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "sess-1.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((directory / "current").stat().st_mode) == 0o600


def test_current_pointer_round_trip(tmp_path: Path) -> None:
    """set_current publishes the pointer that later commands resolve."""
    store = SessionStateStore(tmp_path)
    assert store.current_session_id() is None

    store.save(_state("sess-1"))
    store.set_current("sess-1")
    assert store.current_session_id() == "sess-1"

    store.save(_state("sess-2"))
    store.set_current("sess-2")
    assert store.current_session_id() == "sess-2"


def test_delete_removes_state_archives_and_matching_pointer(tmp_path: Path) -> None:
    """Deleting a session clears every file it owns, including a matching pointer."""
    store = SessionStateStore(tmp_path)
    workspace = WorkspaceCheckpoint(root="/tmp/project")
    store.save(_state(workspace=workspace), base_archive=b"archive-bytes")
    store.set_current("sess-1")

    store.delete("sess-1")

    assert store.load("sess-1") is None
    assert store.current_session_id() is None
    assert list(tmp_path.glob("sess-1*")) == []


def test_delete_keeps_a_pointer_to_another_session(tmp_path: Path) -> None:
    """Deleting a non-current session must not clear the current pointer."""
    store = SessionStateStore(tmp_path)
    store.save(_state("sess-1"))
    store.save(_state("sess-2"))
    store.set_current("sess-2")

    store.delete("sess-1")

    assert store.current_session_id() == "sess-2"


def test_base_archive_round_trip_and_integrity(tmp_path: Path) -> None:
    """The workspace base archive persists content-addressed and verifies on load."""
    store = SessionStateStore(tmp_path)
    saved = store.save(
        _state(workspace=WorkspaceCheckpoint(root="/tmp/project")),
        base_archive=b"archive-bytes",
    )

    assert saved.workspace is not None
    assert saved.workspace.base_sha256
    assert store.load_base_archive(saved) == b"archive-bytes"

    reloaded = store.load("sess-1")
    assert reloaded is not None
    assert reloaded.workspace == saved.workspace


def test_tampered_base_archive_fails_integrity(tmp_path: Path) -> None:
    """A modified checkpoint archive is rejected, never silently used as a base."""
    store = SessionStateStore(tmp_path)
    saved = store.save(
        _state(workspace=WorkspaceCheckpoint(root="/tmp/project")),
        base_archive=b"archive-bytes",
    )
    (archive_path,) = tmp_path.glob("sess-1.workspace-*.tar.gz")
    archive_path.write_bytes(b"tampered")

    with pytest.raises(SessionStateError, match="integrity"):
        store.load_base_archive(saved)


def test_new_base_archive_replaces_the_previous_one(tmp_path: Path) -> None:
    """Checkpoint updates leave exactly one archive on disk (the referenced one)."""
    store = SessionStateStore(tmp_path)
    state = _state(workspace=WorkspaceCheckpoint(root="/tmp/project"))
    first = store.save(state, base_archive=b"first")
    second = store.save(first, base_archive=b"second")

    archives = list(tmp_path.glob("sess-1.workspace-*.tar.gz"))
    assert len(archives) == 1
    assert store.load_base_archive(second) == b"second"


def test_save_base_archive_requires_workspace_metadata(tmp_path: Path) -> None:
    """An archive without checkpoint metadata would be unreachable; refuse it."""
    store = SessionStateStore(tmp_path)
    with pytest.raises(SessionStateError, match="workspace"):
        store.save(_state(), base_archive=b"orphan")


def test_corrupted_state_file_is_an_actionable_error(tmp_path: Path) -> None:
    """A broken state file names its path so the user can remove it."""
    store = SessionStateStore(tmp_path)
    store.save(_state())
    (tmp_path / "sess-1.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(SessionStateError, match="sess-1.json"):
        store.load("sess-1")


def test_unsafe_session_ids_are_rejected(tmp_path: Path) -> None:
    """A session id is a file-name component; traversal never leaves the directory."""
    store = SessionStateStore(tmp_path)
    with pytest.raises(SessionStateError, match="session id"):
        store.load("../escape")
    with pytest.raises(SessionStateError, match="session id"):
        store.set_current("a/b")


def test_symlinked_state_file_is_refused(tmp_path: Path) -> None:
    """A symlink at the state path must never be written through."""
    store = SessionStateStore(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "sess-1.json").symlink_to(outside)

    with pytest.raises(SessionStateError, match="symlink"):
        store.save(_state())


def test_atomic_write_cleanup_preserves_the_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupt racing os.replace must not be masked by the temp-file cleanup."""
    import wmh.cli.session_state as session_state_module

    real_replace = session_state_module.os.replace

    def replace_then_interrupt(src: str, dst: str) -> None:
        real_replace(src, dst)
        raise KeyboardInterrupt

    store = SessionStateStore(tmp_path)
    monkeypatch.setattr(session_state_module.os, "replace", replace_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        store.save(_state())


def test_state_directories_are_created_owner_only_at_every_level(tmp_path: Path) -> None:
    """Every directory level the store creates is 0700 from the start."""
    pre_existing = tmp_path / "home"
    pre_existing.mkdir(mode=0o755)
    directory = pre_existing / ".wmh" / "sessions"
    store = SessionStateStore(directory)

    store.save(_state())

    assert stat.S_IMODE((pre_existing / ".wmh").stat().st_mode) == 0o700
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    # A directory the store did not create keeps its owner-chosen mode.
    assert stat.S_IMODE(pre_existing.stat().st_mode) == 0o755

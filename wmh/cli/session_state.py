# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Persisted local references to detached hosted agent sessions.

``wmh run <agent-id> --detach`` leaves a platform-owned E2B session running
with no local process attached. The CLI remembers how to address it again
(platform URL, agent id, session id) plus the workspace-sync checkpoint in the
user-global WMH state directory (``$WMH_HOME`` or ``~/.wmh``), never inside
the directory being synchronized. Writes are atomic and owner-only; the state
directory is injectable so tests never touch real user state.

The workspace base archive is stored content-addressed next to the JSON state
and referenced by digest: the archive lands first and the state referencing it
lands last, so a crash between the two writes leaves the previous checkpoint
intact instead of a dangling pointer.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from wmh.platform.credentials import wmh_home

if TYPE_CHECKING:
    from pathlib import Path

SESSIONS_DIRNAME = "sessions"
_CURRENT_FILENAME = "current"
# Session ids come from the platform (UUIDs); refuse anything that could
# escape the state directory when used as a file-name component.
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SessionStateError(RuntimeError):
    """Detached-session state on disk is missing, unsafe, or corrupted."""


class WorkspaceCheckpoint(BaseModel):
    """Synchronization checkpoint between one local directory and the session."""

    root: str
    base_sha256: str = ""
    conflicts: tuple[str, ...] = ()
    # A patch applied locally whose acknowledgement did not reach the platform
    # yet; the next command retries it so the server object never leaks.
    pending_ack: str | None = None


class DetachedSessionState(BaseModel):
    """One hosted agent session the CLI can send to, attach to, or end later."""

    api_url: str
    web_url: str | None = None
    agent_id: str
    agent_name: str
    session_id: str
    created_at: str
    cursor: int = 0
    workspace: WorkspaceCheckpoint | None = None


class SessionStateStore:
    """Atomic, owner-only persistence for detached session references."""

    def __init__(self, directory: Path | None = None) -> None:
        """Store state under ``directory`` (default: the user-global WMH home)."""
        self._directory = directory if directory is not None else wmh_home() / SESSIONS_DIRNAME

    # -- session records ---------------------------------------------------------------------

    def save(
        self, state: DetachedSessionState, *, base_archive: bytes | None = None
    ) -> DetachedSessionState:
        """Persist one session's state, optionally with a new workspace base archive.

        Returns:
            The state as persisted (with ``base_sha256`` updated when an
            archive was written).

        Raises:
            SessionStateError: If an archive is given without workspace
                metadata, the id is unsafe, or a state path is a symlink.
        """
        self._ensure_directory()
        session_id = self._validated(state.session_id)
        if base_archive is not None:
            if state.workspace is None:
                msg = "a workspace base archive requires workspace checkpoint metadata"
                raise SessionStateError(msg)
            digest = hashlib.sha256(base_archive, usedforsecurity=False).hexdigest()
            self._write_bytes(self._archive_path(session_id, digest), base_archive)
            state = state.model_copy(
                update={"workspace": state.workspace.model_copy(update={"base_sha256": digest})}
            )
        payload = state.model_dump_json(indent=2)
        self._write_bytes(self._state_path(session_id), payload.encode("utf-8"))
        self._prune_archives(state)
        return state

    def load(self, session_id: str) -> DetachedSessionState | None:
        """Read one session's state; ``None`` when nothing is stored for the id."""
        path = self._state_path(self._validated(session_id))
        if not path.exists():
            return None
        try:
            return DetachedSessionState.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError) as error:
            msg = (
                f"session state at {path} is unreadable; delete it and start a new "
                "session with `wmh run <agent-id> --detach`"
            )
            raise SessionStateError(msg) from error

    def load_base_archive(self, state: DetachedSessionState) -> bytes:
        """Read and integrity-check the persisted workspace base archive."""
        workspace = state.workspace
        if workspace is None or not workspace.base_sha256:
            msg = f"session {state.session_id} has no workspace checkpoint"
            raise SessionStateError(msg)
        path = self._archive_path(self._validated(state.session_id), workspace.base_sha256)
        try:
            content = path.read_bytes()
        except OSError as error:
            msg = (
                f"workspace checkpoint archive is missing at {path}; end the session "
                f"with `wmh run --session {state.session_id} --end` and recover from "
                "the final workspace download"
            )
            raise SessionStateError(msg) from error
        digest = hashlib.sha256(content, usedforsecurity=False).hexdigest()
        if digest != workspace.base_sha256:
            msg = (
                f"workspace checkpoint archive at {path} failed its integrity check; "
                f"delete it and end the session with `wmh run --session "
                f"{state.session_id} --end`"
            )
            raise SessionStateError(msg)
        return content

    def write_recovery_archive(self, session_id: str, content: bytes) -> Path:
        """Save a final workspace archive that could not be synchronized locally.

        The file deliberately survives :meth:`delete`: it is the user's data,
        not session state.
        """
        self._ensure_directory()
        path = self._directory / f"{self._validated(session_id)}.recovered.tar.gz"
        self._write_bytes(path, content)
        return path

    def delete(self, session_id: str) -> None:
        """Remove the state, its checkpoint archives, and a matching current pointer."""
        session_id = self._validated(session_id)
        self._state_path(session_id).unlink(missing_ok=True)
        for archive in self._directory.glob(f"{session_id}.workspace-*.tar.gz"):
            archive.unlink(missing_ok=True)
        if self.current_session_id() == session_id:
            (self._directory / _CURRENT_FILENAME).unlink(missing_ok=True)

    # -- current pointer ---------------------------------------------------------------------

    def set_current(self, session_id: str) -> None:
        """Publish ``session_id`` as the session bare send/attach/end commands use."""
        self._ensure_directory()
        content = f"{self._validated(session_id)}\n".encode()
        self._write_bytes(self._directory / _CURRENT_FILENAME, content)

    def current_session_id(self) -> str | None:
        """The current session id, or ``None`` when no pointer is set."""
        path = self._directory / _CURRENT_FILENAME
        if not path.exists():
            return None
        value = path.read_text(encoding="utf-8").strip()
        return value or None

    # -- internals ---------------------------------------------------------------------------

    def _ensure_directory(self) -> None:
        """Create every missing directory level owner-only from the start.

        ``mkdir(parents=True)`` would create intermediate levels with the
        umask default, leaving a window (and, for ``~/.wmh``, a permanent
        0755) around checkpoint archives that contain the user's source
        tree. A umask can only narrow 0o700, so creating each level with
        that mode closes the race; pre-existing directories are left alone.
        """
        missing: list[Path] = []
        current = self._directory
        while not current.exists():
            missing.append(current)
            parent = current.parent
            if parent == current:
                break
            current = parent
        for directory in reversed(missing):
            with contextlib.suppress(FileExistsError):
                directory.mkdir(mode=0o700)

    def _state_path(self, session_id: str) -> Path:
        return self._directory / f"{session_id}.json"

    def _archive_path(self, session_id: str, digest: str) -> Path:
        return self._directory / f"{session_id}.workspace-{digest[:16]}.tar.gz"

    def _prune_archives(self, state: DetachedSessionState) -> None:
        """Drop archives the just-written state no longer references."""
        keep: str | None = None
        if state.workspace is not None and state.workspace.base_sha256:
            keep = self._archive_path(state.session_id, state.workspace.base_sha256).name
        for candidate in self._directory.glob(f"{state.session_id}.workspace-*.tar.gz"):
            if candidate.name != keep:
                candidate.unlink(missing_ok=True)

    def _validated(self, session_id: str) -> str:
        """Reject ids that are unsafe as file-name components."""
        if _SESSION_ID.fullmatch(session_id) is None:
            msg = f"invalid session id: {session_id!r}"
            raise SessionStateError(msg)
        return session_id

    def _write_bytes(self, path: Path, content: bytes) -> None:
        """Write through a 0600 temporary file and swap into place atomically."""
        if path.is_symlink():
            msg = f"refusing to write session state through the symlink {path}; remove the link"
            raise SessionStateError(msg)
        fd, tmp_name = tempfile.mkstemp(dir=self._directory, prefix=f"{path.name}.")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            os.replace(tmp_name, path)
        except BaseException:
            # The replace may already have consumed the temp file; a missing
            # file must not mask the original exception (e.g. an interrupt).
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

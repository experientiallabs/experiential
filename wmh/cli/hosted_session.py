# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Detached lifecycle and workspace transport for hosted E2B agent sessions.

``wmh run <agent-id> --detach`` starts a normal platform-owned agent session
and returns immediately, remembering it as the current session in WMH user
state. Later ``wmh run --send/--attach/--end`` invocations address that
session (or any accessible session via ``--session <id>``) through the same
authenticated platform protocol the web app uses: durable transcript polling,
queued commands, live workspace patches, and the final workspace handoff.

No local process runs between invocations, so a persisted checkpoint (the
transcript cursor plus the last synchronized workspace snapshot) lets the next
command catch up on hosted workspace patches and upload local edits before
proceeding. Sessions remain ordinary platform records: they stay visible and
controllable from the web application throughout.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer
from rich.console import Console

from wmh.cli.session_state import (
    DetachedSessionState,
    SessionStateError,
    SessionStateStore,
    WorkspaceCheckpoint,
)
from wmh.cli.workspace_sync import (
    WorkspaceSnapshot,
    WorkspaceSyncError,
    advance_snapshot_paths,
    apply_patch_to_snapshot,
    apply_workspace_patch,
    snapshot_from_archive,
    snapshot_workspace,
    sync_workspace,
    write_conflict_archive,
)
from wmh.harness.live_session import SessionEvent
from wmh.harness.workspace_patch import WorkspacePatchError, build_workspace_patch
from wmh.platform.client import PlatformClient, PlatformError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from wmh.cli.workspace_sync import SyncResult
    from wmh.platform.client import (
        RemoteAgentEventPage,
        RemoteAgentSession,
        RemoteAgentSessionEvent,
    )
    from wmh.platform.credentials import PlatformCredentials

_console = Console()

TERMINAL_STATUSES = frozenset({"ended", "failed"})
SessionAction = Literal["send", "attach", "end"]

_POLL_INTERVAL_S = 0.5
_WORKSPACE_SYNC_TICK_S = 1.0
# A session whose driver died keeps answering `running` on the events poll;
# the detail read reconciles it server-side, so probe it occasionally.
_STALE_PROBE_S = 30.0


def patch_revision(event: RemoteAgentSessionEvent) -> str:
    """Extract the patch revision announced by one ``workspace_patch`` event."""
    revision = event.payload.get("revision")
    if not isinstance(revision, str) or not revision:
        raise WorkspaceSyncError("workspace patch event has no revision")
    return revision


class LiveWorkspace:
    """Bidirectional file transport between one local root and a hosted session."""

    def __init__(
        self,
        client: PlatformClient,
        agent_id: str,
        session_id: str,
        root: Path,
        synchronized: WorkspaceSnapshot,
        conflicts: Iterable[str] = (),
    ) -> None:
        """Bind the transport to its synchronized base snapshot and known conflicts."""
        self._client = client
        self._agent_id = agent_id
        self._session_id = session_id
        self.root = root
        self.synchronized = synchronized
        self.conflicts: set[str] = set(conflicts)

    def apply_remote_patch(
        self, revision: str, *, before_ack: Callable[[], None] | None = None
    ) -> None:
        """Download and apply one announced E2B patch, then advance the local base.

        ``before_ack`` runs after the local base advanced but before the patch
        is acknowledged: a detached checkpoint persisted there guarantees an
        acknowledged (hence deleted) patch is never needed again.
        """
        content = self._client.download_agent_workspace_patch(
            self._agent_id, self._session_id, revision
        )
        result = apply_workspace_patch(self.root, content)
        new_conflicts = self._record_conflicts(result.conflicts)
        # Advance the base by base+patch, never by re-reading the directory:
        # a disk snapshot here would absorb not-yet-uploaded local edits into
        # the base, so they would never upload (and the final sync could even
        # delete them). Conflicted paths stay at their base state.
        self.synchronized = apply_patch_to_snapshot(
            self.synchronized, content, conflicts=result.conflicts
        )
        if before_ack is not None:
            before_ack()
        self._client.acknowledge_agent_workspace_patch(self._agent_id, self._session_id, revision)
        if result.applied:
            _console.print(f"[dim]workspace updated ({len(result.applied)} changed paths)[/dim]")
        if new_conflicts:
            paths = ", ".join(new_conflicts)
            _console.print(f"[yellow]workspace sync conflict[/yellow]: {paths}")

    def push_local(self) -> bool:
        """Send local edits made since the last synchronized snapshot.

        Returns:
            Whether the synchronized base advanced (all changes were accepted).
        """
        try:
            current = snapshot_workspace(self.root)
        except WorkspaceSyncError:
            return False
        content = build_workspace_patch(self.synchronized.archive, current.archive)
        if content is None:
            return False
        result = self._client.upload_agent_workspace_patch(
            self._agent_id, self._session_id, content
        )
        new_conflicts = self._record_conflicts(result.conflicts)
        if new_conflicts:
            paths = ", ".join(new_conflicts)
            _console.print(f"[yellow]workspace sync conflict[/yellow]: {paths}")
        if result.conflicts:
            # A conflicted path was rejected by E2B, so ``current`` cannot
            # become the synchronized base for it; accepted sibling paths did
            # land, so they advance individually.
            if result.applied:
                self.synchronized = advance_snapshot_paths(
                    self.synchronized, current, result.applied
                )
            return False
        self.synchronized = current
        return True

    def try_push_local(self) -> bool:
        """Push local edits, tolerating a workspace that cannot accept patches yet.

        A sandbox that is still booting (or winding down) answers 409/503; the
        caller's sync loop retries on its next tick, and an end falls back to
        the conflict-preserving final sync. Anything else still raises.
        """
        try:
            return self.push_local()
        except PlatformError as error:
            if error.status_code not in {409, 503}:
                raise
            _console.print("[dim]local changes will sync once the workspace is running[/dim]")
            return False

    def _record_conflicts(self, conflicts: Iterable[str]) -> list[str]:
        """Track conflicts, returning only ones not already reported."""
        fresh = [path for path in conflicts if path not in self.conflicts]
        self.conflicts.update(conflicts)
        return fresh

    def finalize(self) -> SyncResult:
        """Reconcile the terminal session's final E2B workspace into the root.

        Applies the three-way merge against the last synchronized snapshot,
        preserves conflicting local paths (plus the full result under
        ``.wmh-conflicts/``), and acknowledges the handoff so the platform can
        remove its private archive objects.
        """
        with _console.status("[dim]syncing E2B workspace back...[/dim]", spinner="dots"):
            final_archive = self._client.download_agent_workspace(self._agent_id, self._session_id)
            result = sync_workspace(
                self.root,
                self.synchronized,
                final_archive,
                protected_paths=frozenset(self.conflicts),
            )
        if result.conflicts:
            recovery = write_conflict_archive(self.root, self._session_id, final_archive)
            self._client.acknowledge_agent_workspace(self._agent_id, self._session_id)
            paths = ", ".join(result.conflicts)
            _console.print(
                f"[red]workspace conflicts preserved locally[/red]: {paths}\n"
                f"The full E2B result is saved at [bold]{recovery}[/bold]."
            )
        else:
            self._client.acknowledge_agent_workspace(self._agent_id, self._session_id)
            _console.print(f"[green]workspace synced[/green] ({len(result.applied)} changed paths)")
        return result


class DetachedStartDriver:
    """Start a hosted agent session, persist its reference, and return."""

    def __init__(
        self,
        *,
        client: PlatformClient,
        credentials: PlatformCredentials,
        state_store: SessionStateStore,
        target_id: str,
        name: str,
        jail_root: Path | None,
        task: str | None,
    ) -> None:
        """Store the resolved agent target and optional workspace root."""
        self._client = client
        self._credentials = credentials
        self._store = state_store
        self._target_id = target_id
        self._name = name
        self._jail = jail_root
        self._task = task

    def run(self) -> None:
        """Create the session, persist the reference and checkpoint, and exit."""
        try:
            initial: WorkspaceSnapshot | None = None
            if self._jail is not None:
                with _console.status("[dim]snapshotting local workspace...[/dim]", spinner="dots"):
                    initial = snapshot_workspace(self._jail)
                _console.print(
                    f"[dim]uploading {len(initial.files)} files to the platform "
                    "E2B workspace...[/dim]"
                )
            session = self._client.create_agent_session(
                self._target_id,
                workspace=initial.archive if initial is not None else None,
                instruction=self._task,
            )
            workspace: WorkspaceCheckpoint | None = None
            if self._jail is not None:
                workspace = WorkspaceCheckpoint(root=str(self._jail))
            state = DetachedSessionState(
                api_url=str(self._credentials.api_url),
                web_url=self._credentials.web_url,
                agent_id=self._target_id,
                agent_name=self._name,
                session_id=session.id,
                created_at=datetime.now(tz=UTC).isoformat(),
                workspace=workspace,
            )
            try:
                self._store.save(
                    state, base_archive=initial.archive if initial is not None else None
                )
                self._store.set_current(session.id)
            except SessionStateError as error:
                # The hosted session is already running; the user must get an
                # addressable reference even though the local save failed.
                msg = (
                    f"session {session.id} is running on the platform, but its local "
                    f"reference could not be saved: {error}. Control it with "
                    f"`wmh run --session {session.id} --attach` or end it with "
                    f"`wmh run --session {session.id} --end`"
                )
                raise typer.BadParameter(msg) from error
            _console.print(
                f"[green]detached E2B session started[/green] for [bold]{self._name}[/bold]\n"
                f"  agent    {self._target_id}\n"
                f"  session  {session.id}\n"
                'Send a message with [bold]wmh run -s "..."[/bold], attach with '
                "[bold]wmh run -a[/bold], end with [bold]wmh run --end[/bold]."
            )
        except (WorkspacePatchError, WorkspaceSyncError, SessionStateError) as error:
            raise typer.BadParameter(str(error)) from error
        except PlatformError as error:
            raise typer.BadParameter(str(error)) from error
        finally:
            self._client.close()


class AttachedCommandReader(threading.Thread):
    """Terminal input for an attached session: steer, interrupt, detach, or end."""

    def __init__(self, client: PlatformClient, agent_id: str, session_id: str) -> None:
        """Read stdin on a daemon thread; commands post through the platform."""
        super().__init__(daemon=True)
        self._client = client
        self._agent_id = agent_id
        self._session_id = session_id
        self.detach = threading.Event()
        self.ended = threading.Event()

    def run(self) -> None:
        """Map lines to hosted commands; leaving the terminal detaches, never ends."""
        try:
            for raw in sys.stdin:
                if self.detach.is_set():
                    return
                line = raw.strip()
                if line in {":detach", ":quit", ":q", ":exit"}:
                    self.detach.set()
                    return
                if line == ":end":
                    # A failed end must be reported, never silently converted
                    # into a detach that leaves the session running; after a
                    # successful end the driver streams to the final handoff,
                    # so a following EOF must not look like a detach either.
                    try:
                        self._client.end_agent_session(self._agent_id, self._session_id)
                    except PlatformError as error:
                        _console.print(
                            f"[red]end failed:[/red] {error} "
                            "(retry [bold]:end[/bold], or run `wmh run --end` later)"
                        )
                    else:
                        self.ended.set()
                        return
                elif line == ":stop":
                    self._post("interrupt")
                elif line.startswith(":"):
                    # An unknown command must never reach the agent as chat.
                    _console.print(
                        f"[yellow]unknown command {line}; use :stop, :detach, or :end[/yellow]"
                    )
                elif line:
                    self._post("user_message", text=line)
        except OSError:
            pass
        finally:
            # Closed stdin means the terminal went away, not that the hosted
            # session should end. Ending stays explicit (:end or --end); after
            # one, the driver keeps streaming to the final handoff.
            if not self.ended.is_set():
                self.detach.set()

    def _post(self, kind: str, *, text: str | None = None) -> None:
        """Post one command; a transient failure warns and keeps the reader alive."""
        try:
            self._client.post_agent_session_command(
                self._agent_id, self._session_id, kind, text=text
            )
        except PlatformError as error:
            _console.print(f"[red]{kind} failed:[/red] {error} (still attached; try again)")


@dataclass
class _Stream:
    """Mutable streaming state threaded through the detached event loop."""

    state: DetachedSessionState
    persisted: bool
    workspace: LiveWorkspace | None
    render: bool
    cursor: int = 0
    pending_ack: str | None = None
    pending_text: str | None = None
    message_seen: bool = False
    turn_idle: bool = False
    foreign_patch_noted: bool = False


class DetachedCommandDriver:
    """Send to, attach to, or end one hosted session from its stored reference."""

    def __init__(
        self,
        *,
        client: PlatformClient,
        credentials: PlatformCredentials,
        state_store: SessionStateStore,
        action: SessionAction,
        text: str | None,
        session_override: str | None,
        sink: Callable[[SessionEvent], None],
    ) -> None:
        """Bind one action to the resolved credentials, state store, and renderer."""
        self._client = client
        self._credentials = credentials
        self._store = state_store
        self._action = action
        self._text = text
        self._session_override = session_override
        self._sink = sink
        self._interrupts = 0
        self._persisted_snapshot: WorkspaceSnapshot | None = None
        self._salvage_reason: str | None = None

    def run(self) -> None:
        """Resolve the session, catch up, and run the requested action."""
        try:
            self._run()
        except (WorkspacePatchError, WorkspaceSyncError, SessionStateError) as error:
            raise typer.BadParameter(str(error)) from error
        except PlatformError as error:
            raise typer.BadParameter(str(error)) from error
        finally:
            self._client.close()

    # -- resolution --------------------------------------------------------------------------

    def _run(self) -> None:
        state, persisted, remote = self._resolve()
        if remote.status in TERMINAL_STATUSES:
            self._handle_already_terminal(state, persisted, remote)
            return
        workspace = self._load_workspace(state, persisted)
        stream = _Stream(
            state=state,
            persisted=persisted,
            workspace=workspace,
            render=persisted,
            cursor=state.cursor,
            pending_ack=state.workspace.pending_ack if state.workspace is not None else None,
        )
        self._retry_pending_ack(stream)
        match self._action:
            case "send":
                self._send(stream)
            case "attach":
                self._attach(stream)
            case "end":
                self._end(stream)

    def _resolve(self) -> tuple[DetachedSessionState, bool, RemoteAgentSession]:
        """Resolve the addressed session to state, persistence, and remote record."""
        api_url = str(self._credentials.api_url)
        override = self._session_override
        if override is not None:
            stored = self._store.load(override)
            if stored is None:
                return self._resolve_remote(override, api_url)
            state = stored
        else:
            current = self._store.current_session_id()
            if current is None:
                raise typer.BadParameter(
                    "no current session: start one with `wmh run <agent-id> --detach`, "
                    "or address one with --session <session-id>"
                )
            loaded = self._store.load(current)
            if loaded is None:
                raise typer.BadParameter(
                    f"the current session pointer references {current} but its state is "
                    "gone; start a new session with `wmh run <agent-id> --detach`"
                )
            state = loaded
        if state.api_url != api_url:
            stored_home = state.web_url or state.api_url
            active_home = self._credentials.web_url or api_url
            raise typer.BadParameter(
                f"session {state.session_id} belongs to {stored_home}, but this login "
                f"points at {active_home}; run `wmh login --url {stored_home}` to switch, "
                "or start a new session here with `wmh run <agent-id> --detach`"
            )
        try:
            remote = self._client.get_agent_session(state.agent_id, state.session_id)
        except PlatformError as error:
            if error.status_code == 404:
                raise typer.BadParameter(
                    f"session {state.session_id} was not found on "
                    f"{state.web_url or state.api_url}; it may have been removed, or "
                    "this login may lack access (check `wmh status`)"
                ) from error
            raise
        return state, True, remote

    def _resolve_remote(
        self, session_id: str, api_url: str
    ) -> tuple[DetachedSessionState, bool, RemoteAgentSession]:
        """Resolve a session id with no local state through the platform."""
        try:
            remote = self._client.resolve_agent_session(session_id)
        except PlatformError as error:
            if error.status_code == 404:
                raise typer.BadParameter(
                    f"session {session_id} was not found on "
                    f"{self._credentials.web_url or api_url}; check --session and that "
                    "your login (`wmh status`) can access it"
                ) from error
            raise
        name = remote.agent_id
        with contextlib.suppress(PlatformError):
            target = self._client.resolve_run_target(remote.agent_id)
            name = target.display_name or target.name
        state = DetachedSessionState(
            api_url=api_url,
            web_url=self._credentials.web_url,
            agent_id=remote.agent_id,
            agent_name=name,
            session_id=session_id,
            created_at=datetime.now(tz=UTC).isoformat(),
        )
        return state, False, remote

    def _handle_already_terminal(
        self, state: DetachedSessionState, persisted: bool, remote: RemoteAgentSession
    ) -> None:
        """Reconcile (--end) or explain (send/attach) an already-finished session."""
        if self._action != "end":
            detail = (
                "reconcile its final workspace and clear it"
                if persisted and state.workspace is not None
                else "clear it"
            )
            raise typer.BadParameter(
                f"session {state.session_id} is already {remote.status}; run "
                f"`wmh run --session {state.session_id} --end` to {detail}, or start a "
                f"new session with `wmh run {state.agent_id} --detach`"
            )
        workspace = self._load_workspace(state, persisted)
        stream = _Stream(
            state=state,
            persisted=persisted,
            workspace=workspace,
            render=False,
            cursor=state.cursor,
            pending_ack=state.workspace.pending_ack if state.workspace is not None else None,
        )
        self._retry_pending_ack(stream)
        self._finish_terminal(stream)

    def _load_workspace(self, state: DetachedSessionState, persisted: bool) -> LiveWorkspace | None:
        """Rehydrate the workspace transport from the persisted checkpoint."""
        if not persisted or state.workspace is None:
            return None
        root = Path(state.workspace.root)
        if not root.is_dir():
            if self._action == "end":
                self._salvage_reason = f"the synchronized workspace {root} no longer exists"
                return None
            raise typer.BadParameter(
                f"the synchronized workspace {root} no longer exists; restore it, or run "
                f"`wmh run --session {state.session_id} --end` to save the final "
                "workspace without a local sync"
            )
        try:
            base = self._store.load_base_archive(state)
        except SessionStateError as error:
            # `--end` still completes the handoff without a local sync; the
            # stored error messages point other actions at exactly that.
            if self._action == "end":
                self._salvage_reason = str(error)
                return None
            raise
        synchronized = snapshot_from_archive(base)
        self._persisted_snapshot = synchronized
        return LiveWorkspace(
            self._client,
            state.agent_id,
            state.session_id,
            root,
            synchronized,
            conflicts=state.workspace.conflicts,
        )

    # -- actions -----------------------------------------------------------------------------

    def _send(self, stream: _Stream) -> None:
        """Catch up, deliver one message, and stream its turn until idle."""
        text = self._text if self._text is not None else ""
        self._push_workspace(stream)
        if self._catch_up(stream) == "terminal":
            self._finish_terminal(
                stream, failure_note="the session ended before the message could be sent"
            )
            return
        self._client.post_agent_session_command(
            stream.state.agent_id, stream.state.session_id, "user_message", text=text
        )
        stream.pending_text = text
        stream.render = True
        outcome = self._stream_until(stream, stop=lambda: stream.turn_idle)
        if outcome == "terminal":
            self._finish_terminal(stream)
            return
        self._persist(stream, stream.cursor)
        if outcome == "detached":
            _console.print("[dim]detached; the session stays alive[/dim]")
            return
        _console.print(f"[dim]turn finished; session {stream.state.session_id} stays alive[/dim]")

    def _attach(self, stream: _Stream) -> None:
        """Catch up, then stream interactively until the user detaches or ends."""
        self._push_workspace(stream)
        if self._catch_up(stream) == "terminal":
            self._finish_terminal(stream)
            return
        stream.render = True
        reader = AttachedCommandReader(self._client, stream.state.agent_id, stream.state.session_id)
        reader.start()
        _console.print(
            f"[green]attached[/green] to [bold]{stream.state.agent_name}[/bold] "
            f"({stream.state.session_id}). Type to steer, [bold]:stop[/bold] to interrupt, "
            "[bold]:detach[/bold] to leave it running, [bold]:end[/bold] to end it."
        )
        outcome = self._stream_until(stream, stop=reader.detach.is_set)
        if outcome == "terminal":
            self._finish_terminal(stream)
            return
        self._persist(stream, stream.cursor)
        _console.print("[dim]detached; the session stays alive[/dim]")

    def _end(self, stream: _Stream) -> None:
        """Catch up, push local edits, end the session, and run the final sync."""
        self._push_workspace(stream)
        if self._catch_up(stream) != "terminal":
            remote = self._client.end_agent_session(stream.state.agent_id, stream.state.session_id)
            if remote.status not in TERMINAL_STATUSES:
                outcome = self._stream_until(stream, stop=lambda: False)
                if outcome == "detached":
                    self._persist(stream, stream.cursor)
                    _console.print(
                        "[yellow]end requested; leaving before the final sync (rerun "
                        "`wmh run --end` to finish)[/yellow]"
                    )
                    raise typer.Exit(code=1)
        self._finish_terminal(stream)

    def _finish_terminal(self, stream: _Stream, *, failure_note: str | None = None) -> None:
        """Run the final workspace handoff, clear local state, and report the outcome."""
        conflicted = False
        if stream.workspace is not None:
            try:
                result = stream.workspace.finalize()
                conflicted = bool(result.conflicts)
            except PlatformError as error:
                if error.status_code != 404:
                    # Keep local state so `wmh run --end` can retry the handoff.
                    raise
                _console.print(
                    "[yellow]no final workspace archive is available; it may already "
                    "have been synchronized[/yellow]"
                )
        elif (
            stream.persisted
            and stream.state.workspace is not None
            and self._salvage_reason is not None
        ):
            self._salvage_final_workspace(stream)
        terminal: RemoteAgentSession | None
        try:
            terminal = self._client.get_agent_session(
                stream.state.agent_id, stream.state.session_id
            )
        except PlatformError as error:
            # The record (or its agent) can vanish between the handoff and
            # this read; local state must still be cleared, or it is stuck
            # with no CLI path to remove it.
            if error.status_code != 404:
                raise
            terminal = None
        if stream.persisted:
            self._store.delete(stream.state.session_id)
        if failure_note is not None:
            _console.print(f"[red]{failure_note}[/red]")
        if terminal is not None and terminal.status == "failed":
            _console.print(f"[red]session failed: {terminal.error or 'unknown error'}[/red]")
            raise typer.Exit(code=1)
        detail = (
            terminal.ended_reason or terminal.status
            if terminal is not None
            else "no longer visible on the platform"
        )
        _console.print(f"[dim]session ended ({detail})[/dim]")
        if failure_note is not None:
            raise typer.Exit(code=1)
        if conflicted:
            raise typer.Exit(code=2)

    def _salvage_final_workspace(self, stream: _Stream) -> None:
        """Save the final archive without a local sync when the checkpoint is unusable.

        The handoff is still acknowledged so the platform can remove its
        private archive object; the user's data lands as a recovery archive
        (under the workspace root when it exists, else in WMH state, where it
        survives the state cleanup).
        """
        state = stream.state
        try:
            content = self._client.download_agent_workspace(state.agent_id, state.session_id)
        except PlatformError as error:
            if error.status_code != 404:
                # Keep local state so `wmh run --end` can retry the handoff.
                raise
            _console.print(
                "[yellow]no final workspace archive is available; it may already "
                "have been synchronized[/yellow]"
            )
            return
        workspace_state = state.workspace
        root = Path(workspace_state.root) if workspace_state is not None else None
        if root is not None and root.is_dir():
            recovery = write_conflict_archive(root, state.session_id, content)
        else:
            recovery = self._store.write_recovery_archive(state.session_id, content)
        self._client.acknowledge_agent_workspace(state.agent_id, state.session_id)
        _console.print(
            f"[yellow]final workspace saved without a local sync "
            f"({self._salvage_reason}).[/yellow]\n"
            f"The full E2B result is at [bold]{recovery}[/bold]."
        )

    def _retry_pending_ack(self, stream: _Stream) -> None:
        """Deliver a patch acknowledgement a previous invocation could not send."""
        workspace_state = stream.state.workspace
        if not stream.persisted or workspace_state is None or workspace_state.pending_ack is None:
            return
        revision = workspace_state.pending_ack
        try:
            self._client.acknowledge_agent_workspace_patch(
                stream.state.agent_id, stream.state.session_id, revision
            )
        except PlatformError as error:
            # 404 means the object is already gone (acknowledged after all, or
            # removed with the session); anything else stays retryable.
            if error.status_code != 404:
                raise
        stream.pending_ack = None
        self._persist(stream, stream.cursor)

    # -- event loop --------------------------------------------------------------------------

    def _catch_up(self, stream: _Stream) -> str:
        """Apply everything durable that happened while no CLI process ran.

        Returns:
            ``"terminal"`` when the session finished, otherwise ``"ready"``.
        """
        while True:
            page = self._poll(stream)
            if page.status in TERMINAL_STATUSES:
                return "terminal"
            if not page.events:
                return "ready"

    def _stream_until(self, stream: _Stream, *, stop: Callable[[], bool]) -> str:
        """Poll, render, and synchronize until ``stop``, terminal state, or detach.

        Returns:
            ``"terminal"``, ``"stopped"``, or ``"detached"`` (double Ctrl-C).
        """
        last_push = time.monotonic()
        last_probe = time.monotonic()
        while True:
            try:
                page = self._poll(stream)
                if page.status in TERMINAL_STATUSES:
                    return "terminal"
                if stop():
                    return "stopped"
                now = time.monotonic()
                if stream.workspace is not None and now - last_push >= _WORKSPACE_SYNC_TICK_S:
                    self._push_workspace(stream)
                    last_push = now
                if now - last_probe >= _STALE_PROBE_S:
                    # The detail read lazily reconciles a dead driver so the
                    # next events poll reports the truthful terminal status.
                    with contextlib.suppress(PlatformError):
                        self._client.get_agent_session(
                            stream.state.agent_id, stream.state.session_id
                        )
                    last_probe = now
                time.sleep(_POLL_INTERVAL_S)
            except KeyboardInterrupt:
                self._interrupts += 1
                if self._interrupts >= 2:
                    return "detached"
                # The next Ctrl-C frequently lands inside this handler (the
                # print or the HTTP post); it must detach, not crash out.
                try:
                    _console.print("\n[yellow]interrupting (press Ctrl-C again to detach)[/yellow]")
                    with contextlib.suppress(PlatformError):
                        self._client.post_agent_session_command(
                            stream.state.agent_id, stream.state.session_id, "interrupt"
                        )
                except KeyboardInterrupt:
                    return "detached"

    def _poll(self, stream: _Stream) -> RemoteAgentEventPage:
        """Fetch one page after the cursor, process it, and persist the checkpoint."""
        page = self._client.list_agent_session_events(
            stream.state.agent_id, stream.state.session_id, after=stream.cursor
        )
        for event in page.events:
            self._handle_event(stream, event)
            # Advance per event, not only per page: a Ctrl-C landing mid-page
            # must resume after the events already processed (re-fetching a
            # patch event whose object was acknowledged would 404).
            stream.cursor = event.seq
        stream.cursor = page.last_seq
        self._persist(stream, page.last_seq)
        return page

    def _handle_event(self, stream: _Stream, event: RemoteAgentSessionEvent) -> None:
        """Apply one durable event: workspace transport, turn tracking, rendering."""
        if event.kind == "workspace_patch":
            if stream.workspace is None:
                if not stream.foreign_patch_noted:
                    _console.print(
                        "[dim](workspace patches are synchronized by the launching CLI; "
                        "skipping)[/dim]"
                    )
                    stream.foreign_patch_noted = True
                return
            revision = patch_revision(event)

            def checkpoint_before_ack() -> None:
                stream.cursor = event.seq
                stream.pending_ack = revision
                self._persist(stream, event.seq)

            stream.workspace.apply_remote_patch(revision, before_ack=checkpoint_before_ack)
            stream.pending_ack = None
            self._persist(stream, stream.cursor)
            return
        if event.kind == "status":
            detail = event.payload.get("message") or event.payload.get("status")
            if detail and stream.render:
                _console.print(f"[dim]({detail})[/dim]")
            return
        if event.kind == "user_message" and stream.pending_text is not None:
            # Turn attribution matches the echoed text among events after the
            # send-time cursor. The command API cannot correlate a command id
            # to its transcript echo, so an identical message posted by
            # another actor in the same window can end the stream one turn
            # early; the session itself is unaffected (a known approximation).
            if event.payload.get("text") == stream.pending_text:
                stream.message_seen = True
        if event.kind == "state" and stream.message_seen and event.payload.get("status") == "idle":
            stream.turn_idle = True
        if stream.render:
            self._sink(SessionEvent(kind=event.kind, payload=event.payload))

    # -- checkpointing -----------------------------------------------------------------------

    def _push_workspace(self, stream: _Stream) -> None:
        """Upload local edits made since the checkpoint, then persist it."""
        if stream.workspace is None:
            return
        stream.workspace.try_push_local()
        self._persist(stream, stream.cursor)

    def _persist(self, stream: _Stream, cursor: int) -> None:
        """Advance the durable checkpoint (cursor, conflicts, base snapshot)."""
        if not stream.persisted:
            return
        state = stream.state
        workspace_state = state.workspace
        archive: bytes | None = None
        if workspace_state is not None:
            update: dict[str, tuple[str, ...] | str | None] = {"pending_ack": stream.pending_ack}
            if stream.workspace is not None:
                update["conflicts"] = tuple(sorted(stream.workspace.conflicts))
                if stream.workspace.synchronized is not self._persisted_snapshot:
                    archive = stream.workspace.synchronized.archive
            workspace_state = workspace_state.model_copy(update=update)
        if cursor == state.cursor and workspace_state == state.workspace and archive is None:
            return
        updated = state.model_copy(update={"cursor": cursor, "workspace": workspace_state})
        stream.state = self._store.save(updated, base_archive=archive)
        if stream.workspace is not None:
            self._persisted_snapshot = stream.workspace.synchronized

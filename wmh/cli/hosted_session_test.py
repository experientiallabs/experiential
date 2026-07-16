# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for detached hosted-session flows: start, send, attach, end, catch-up."""

from __future__ import annotations

import io
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import typer

import wmh.cli.hosted_session as mod
from wmh.cli.session_state import DetachedSessionState, SessionStateStore, WorkspaceCheckpoint
from wmh.cli.workspace_sync import FileState, snapshot_from_archive, snapshot_workspace
from wmh.harness.workspace_patch import build_workspace_patch
from wmh.platform.client import (
    PlatformError,
    RemoteAgentEventPage,
    RemoteAgentSession,
    RemoteAgentSessionEvent,
    WorkspacePatchResult,
)
from wmh.platform.credentials import PlatformCredentials

if TYPE_CHECKING:
    from wmh.core.types import JsonValue

API_URL = "https://api.test"
WEB_URL = "https://platform.test"


def _credentials() -> PlatformCredentials:
    return PlatformCredentials(api_url=API_URL, web_url=WEB_URL, token="xpl_secret")


def _session(status: str = "running", *, workspace_sync: bool = False) -> RemoteAgentSession:
    return RemoteAgentSession(
        id="sess-1",
        agent_id="agent-1",
        status=status,
        workspace_sync=workspace_sync,
        launched_from="cli",
    )


def _event(seq: int, kind: str, **payload: JsonValue) -> RemoteAgentSessionEvent:
    return RemoteAgentSessionEvent.model_validate(
        {"seq": seq, "kind": kind, "payload": dict(payload)}
    )


def _page(
    events: list[RemoteAgentSessionEvent], last_seq: int, status: str
) -> RemoteAgentEventPage:
    return RemoteAgentEventPage(events=events, last_seq=last_seq, status=status)


def _archive(files: dict[str, bytes]) -> bytes:
    """A regular-file-only gzip tar simulating a hosted workspace archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class _FakeClient:
    """A scriptable platform client covering the hosted-session protocol."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.commands: list[tuple[str, str | None]] = []
        self.end_calls: list[tuple[str, str]] = []
        self.pages: list[RemoteAgentEventPage] = []
        self.session_states: list[RemoteAgentSession] = [_session()]
        self.resolved: RemoteAgentSession | None = None
        self.end_result: RemoteAgentSession | None = None
        self.created: RemoteAgentSession | None = None
        self.created_workspace: bytes | None = None
        self.created_instruction: str | None = None
        self.patches: dict[str, bytes] = {}
        self.patch_acks: list[str] = []
        self.upload_result = WorkspacePatchResult(applied=[], conflicts=[])
        self.uploaded_patches: list[bytes] = []
        self.final_archive: bytes | None = None
        self.final_acked = False
        self.closed = False

    def create_agent_session(
        self, agent_id: str, *, workspace: bytes | None, instruction: str | None = None
    ) -> RemoteAgentSession:
        self.calls.append(f"create {agent_id}")
        self.created_workspace = workspace
        self.created_instruction = instruction
        created = self.created or _session("starting")
        return created

    def get_agent_session(self, agent_id: str, session_id: str) -> RemoteAgentSession:
        self.calls.append(f"get {agent_id}/{session_id}")
        if len(self.session_states) > 1:
            return self.session_states.pop(0)
        return self.session_states[0]

    def resolve_agent_session(self, session_id: str) -> RemoteAgentSession:
        self.calls.append(f"resolve {session_id}")
        if self.resolved is None:
            raise PlatformError(f"Session not found: {session_id}", status_code=404)
        return self.resolved

    def resolve_run_target(self, target_id: str) -> object:
        self.calls.append(f"target {target_id}")
        return SimpleNamespace(display_name=None, name="remote-agent")

    def list_agent_session_events(
        self, agent_id: str, session_id: str, *, after: int
    ) -> RemoteAgentEventPage:
        self.calls.append(f"events after={after}")
        if self.pages:
            return self.pages.pop(0)
        return _page([], after, self.session_states[-1].status)

    def post_agent_session_command(
        self, agent_id: str, session_id: str, kind: str, *, text: str | None = None
    ) -> None:
        self.calls.append(f"command:{kind}")
        self.commands.append((kind, text))

    def end_agent_session(self, agent_id: str, session_id: str) -> RemoteAgentSession:
        self.calls.append(f"end {agent_id}/{session_id}")
        self.end_calls.append((agent_id, session_id))
        return self.end_result or _session("ending")

    def download_agent_workspace_patch(
        self, agent_id: str, session_id: str, revision: str
    ) -> bytes:
        self.calls.append(f"patch:{revision}")
        return self.patches[revision]

    def acknowledge_agent_workspace_patch(
        self, agent_id: str, session_id: str, revision: str
    ) -> None:
        self.calls.append(f"patch-ack:{revision}")
        self.patch_acks.append(revision)

    def upload_agent_workspace_patch(
        self, agent_id: str, session_id: str, content: bytes
    ) -> WorkspacePatchResult:
        self.calls.append("upload_patch")
        self.uploaded_patches.append(content)
        return self.upload_result

    def download_agent_workspace(self, agent_id: str, session_id: str) -> bytes:
        self.calls.append("final_download")
        if self.final_archive is None:
            raise PlatformError("workspace not found", status_code=404)
        return self.final_archive

    def acknowledge_agent_workspace(self, agent_id: str, session_id: str) -> None:
        self.calls.append("final_ack")
        self.final_acked = True

    def close(self) -> None:
        self.closed = True


def _store_with_session(
    tmp_path: Path,
    *,
    workspace_root: Path | None = None,
    base_archive: bytes | None = None,
) -> SessionStateStore:
    """A state store holding sess-1 as the current session."""
    store = SessionStateStore(tmp_path / "state")
    workspace = None
    if workspace_root is not None:
        workspace = WorkspaceCheckpoint(root=str(workspace_root))
    state = DetachedSessionState(
        api_url=API_URL,
        web_url=WEB_URL,
        agent_id="agent-1",
        agent_name="Agent",
        session_id="sess-1",
        created_at="2026-07-15T00:00:00+00:00",
        workspace=workspace,
    )
    store.save(state, base_archive=base_archive)
    store.set_current("sess-1")
    return store


def _command_driver(
    client: _FakeClient,
    store: SessionStateStore,
    *,
    action: mod.SessionAction,
    text: str | None = None,
    session_override: str | None = None,
    sink: list[object] | None = None,
) -> mod.DetachedCommandDriver:
    events = sink if sink is not None else []
    return mod.DetachedCommandDriver(
        client=cast("mod.PlatformClient", client),
        credentials=_credentials(),
        state_store=store,
        action=action,
        text=text,
        session_override=session_override,
        sink=events.append,
    )


# -- LiveWorkspace ---------------------------------------------------------------------------


def test_conflicted_local_push_does_not_advance_the_base(tmp_path: Path) -> None:
    """A rejected same-file edit remains outside the platform-accepted snapshot."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(tmp_path)
    (tmp_path / "answer.txt").write_text("local", encoding="utf-8")
    client = _FakeClient()
    client.upload_result = WorkspacePatchResult(applied=[], conflicts=["answer.txt"])
    workspace = mod.LiveWorkspace(
        cast("mod.PlatformClient", client), "agent-1", "sess-1", tmp_path, base
    )

    changed = workspace.push_local()

    assert not changed
    assert workspace.synchronized is base
    assert workspace.conflicts == {"answer.txt"}


def test_accepted_local_push_advances_the_base(tmp_path: Path) -> None:
    """An accepted patch makes the current local snapshot the new synchronized base."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(tmp_path)
    (tmp_path / "answer.txt").write_text("local", encoding="utf-8")
    client = _FakeClient()
    workspace = mod.LiveWorkspace(
        cast("mod.PlatformClient", client), "agent-1", "sess-1", tmp_path, base
    )

    changed = workspace.push_local()

    assert changed
    assert len(client.uploaded_patches) == 1
    assert workspace.synchronized.files == snapshot_workspace(tmp_path).files


def test_remote_patch_applies_locally_and_checkpoints_before_ack(tmp_path: Path) -> None:
    """The durable checkpoint lands between local apply and the remote acknowledgement."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(tmp_path)
    patch = build_workspace_patch(base.archive, _archive({"answer.txt": b"during"}))
    assert patch is not None
    client = _FakeClient()
    client.patches["patch-1"] = patch
    workspace = mod.LiveWorkspace(
        cast("mod.PlatformClient", client), "agent-1", "sess-1", tmp_path, base
    )
    order: list[str] = []

    workspace.apply_remote_patch("patch-1", before_ack=lambda: order.append("checkpoint"))
    order.extend(f"ack:{revision}" for revision in client.patch_acks)

    assert order == ["checkpoint", "ack:patch-1"]
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "during"
    assert workspace.synchronized.files == snapshot_workspace(tmp_path).files


def test_finalize_preserves_conflicts_and_acknowledges(tmp_path: Path) -> None:
    """Final sync keeps conflicting local work, saves the recovery archive, and acks."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(tmp_path)
    (tmp_path / "answer.txt").write_text("local", encoding="utf-8")
    client = _FakeClient()
    client.final_archive = _archive({"answer.txt": b"remote"})
    workspace = mod.LiveWorkspace(
        cast("mod.PlatformClient", client), "agent-1", "sess-1", tmp_path, base
    )

    result = workspace.finalize()

    assert result.conflicts == ("answer.txt",)
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "local"
    assert (tmp_path / ".wmh-conflicts" / "sess-1.tar.gz").is_file()
    assert client.final_acked


# -- DetachedStartDriver ---------------------------------------------------------------------


def test_detached_start_persists_reference_and_returns(tmp_path: Path) -> None:
    """--detach creates a normal hosted session, records it, and never ends it."""
    store = SessionStateStore(tmp_path / "state")
    client = _FakeClient()
    mod.DetachedStartDriver(
        client=cast("mod.PlatformClient", client),
        credentials=_credentials(),
        state_store=store,
        target_id="agent-1",
        name="Agent",
        jail_root=None,
        task="open task",
    ).run()

    state = store.load("sess-1")
    assert state is not None
    assert state.api_url == API_URL
    assert state.web_url == WEB_URL
    assert state.agent_id == "agent-1"
    assert state.cursor == 0
    assert state.workspace is None
    assert store.current_session_id() == "sess-1"
    assert client.created_instruction == "open task"
    assert client.created_workspace is None
    assert client.commands == []
    assert client.end_calls == []
    assert client.closed


def test_detached_start_uploads_workspace_and_checkpoints_the_base(tmp_path: Path) -> None:
    """-u at detach time uploads the snapshot and persists it as the sync base."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    store = SessionStateStore(tmp_path / "state")
    client = _FakeClient()

    mod.DetachedStartDriver(
        client=cast("mod.PlatformClient", client),
        credentials=_credentials(),
        state_store=store,
        target_id="agent-1",
        name="Agent",
        jail_root=root,
        task=None,
    ).run()

    state = store.load("sess-1")
    assert state is not None
    assert state.workspace is not None
    assert state.workspace.root == str(root)
    assert client.created_workspace is not None
    assert store.load_base_archive(state) == client.created_workspace
    assert snapshot_from_archive(client.created_workspace).files == snapshot_workspace(root).files


# -- DetachedCommandDriver: send ---------------------------------------------------------------


def test_send_streams_one_turn_and_leaves_the_session_alive(tmp_path: Path) -> None:
    """--send posts one user_message, streams to idle, and exits without ending."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.pages = [
        _page([], 0, "running"),
        _page(
            [
                _event(1, "user_message", text="Do this task"),
                _event(2, "state", status="running"),
                _event(3, "assistant_message", text="done"),
                _event(4, "state", status="idle"),
            ],
            4,
            "running",
        ),
    ]
    rendered: list[object] = []

    _command_driver(client, store, action="send", text="Do this task", sink=rendered).run()

    assert client.commands == [("user_message", "Do this task")]
    assert client.end_calls == []
    state = store.load("sess-1")
    assert state is not None
    assert state.cursor == 4
    assert any(getattr(event, "kind", None) == "assistant_message" for event in rendered)
    assert client.closed


def test_send_ignores_idle_left_over_from_a_previous_turn(tmp_path: Path) -> None:
    """Turn completion counts only after the sent message appears in the transcript."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.pages = [
        _page([], 0, "running"),
        _page([_event(1, "state", status="idle")], 1, "running"),
        _page(
            [_event(2, "user_message", text="hi"), _event(3, "state", status="idle")],
            3,
            "running",
        ),
    ]

    _command_driver(client, store, action="send", text="hi").run()

    state = store.load("sess-1")
    assert state is not None
    assert state.cursor == 3


def test_send_catches_up_remote_patches_before_messaging(tmp_path: Path) -> None:
    """Patches hosted while detached land locally (and are acked) before the send."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(root)
    store = _store_with_session(tmp_path, workspace_root=root, base_archive=base.archive)
    patch = build_workspace_patch(base.archive, _archive({"answer.txt": b"during"}))
    assert patch is not None
    client = _FakeClient()
    client.session_states = [_session(workspace_sync=True)]
    client.patches["patch-1"] = patch
    client.pages = [
        _page([_event(1, "workspace_patch", revision="patch-1")], 1, "running"),
        _page([], 1, "running"),
        _page(
            [_event(2, "user_message", text="hi"), _event(3, "state", status="idle")],
            3,
            "running",
        ),
    ]

    _command_driver(client, store, action="send", text="hi").run()

    assert (root / "answer.txt").read_text(encoding="utf-8") == "during"
    assert client.patch_acks == ["patch-1"]
    patch_index = client.calls.index("patch:patch-1")
    command_index = client.calls.index("command:user_message")
    assert patch_index < command_index
    state = store.load("sess-1")
    assert state is not None
    assert state.cursor == 3
    assert snapshot_from_archive(store.load_base_archive(state)).files == base_files(root)


def base_files(root: Path) -> dict[str, FileState]:
    """The current on-disk manifest, for checkpoint equality assertions."""
    return dict(snapshot_workspace(root).files)


def test_send_uploads_local_changes_before_the_message(tmp_path: Path) -> None:
    """Local edits made while detached reach the sandbox before the agent reads the task."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(root)
    store = _store_with_session(tmp_path, workspace_root=root, base_archive=base.archive)
    (root / "answer.txt").write_text("edited while detached", encoding="utf-8")
    client = _FakeClient()
    client.session_states = [_session(workspace_sync=True)]
    client.pages = [
        _page([], 0, "running"),
        _page(
            [_event(1, "user_message", text="hi"), _event(2, "state", status="idle")],
            2,
            "running",
        ),
    ]

    _command_driver(client, store, action="send", text="hi").run()

    assert len(client.uploaded_patches) == 1
    assert client.calls.index("upload_patch") < client.calls.index("command:user_message")
    state = store.load("sess-1")
    assert state is not None
    assert snapshot_from_archive(store.load_base_archive(state)).files == base_files(root)


def test_send_to_terminal_session_is_actionable(tmp_path: Path) -> None:
    """A dead current session tells the user how to reconcile it, keeping state."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.session_states = [_session("ended")]

    with pytest.raises(typer.BadParameter, match="--end"):
        _command_driver(client, store, action="send", text="hi").run()

    assert store.load("sess-1") is not None
    assert client.commands == []


def test_send_without_current_session_is_actionable(tmp_path: Path) -> None:
    """No stored session points the user at --detach and --session."""
    store = SessionStateStore(tmp_path / "state")
    client = _FakeClient()

    with pytest.raises(typer.BadParameter, match="--detach"):
        _command_driver(client, store, action="send", text="hi").run()


def test_dangling_current_pointer_is_actionable(tmp_path: Path) -> None:
    """A pointer whose state file is gone explains how to start over."""
    store = SessionStateStore(tmp_path / "state")
    store.set_current("sess-ghost")
    client = _FakeClient()

    with pytest.raises(typer.BadParameter, match="--detach"):
        _command_driver(client, store, action="send", text="hi").run()


def test_platform_url_mismatch_is_actionable(tmp_path: Path) -> None:
    """A session from another platform names both URLs and the fix."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    driver = mod.DetachedCommandDriver(
        client=cast("mod.PlatformClient", client),
        credentials=PlatformCredentials(
            api_url="https://api.elsewhere", web_url="https://elsewhere.test", token="xpl_2"
        ),
        state_store=store,
        action="send",
        text="hi",
        session_override=None,
        sink=lambda _event: None,
    )

    with pytest.raises(typer.BadParameter, match="wmh login"):
        driver.run()


def test_session_override_resolves_remotely_when_unknown_locally(tmp_path: Path) -> None:
    """--session with a foreign id uses the platform resolution route, ephemerally."""
    store = SessionStateStore(tmp_path / "state")
    client = _FakeClient()
    client.resolved = RemoteAgentSession(
        id="sess-9",
        agent_id="agent-2",
        status="running",
        workspace_sync=False,
        launched_from="web",
    )
    client.pages = [
        _page([], 0, "running"),
        _page(
            [_event(1, "user_message", text="hi"), _event(2, "state", status="idle")],
            2,
            "running",
        ),
    ]

    _command_driver(client, store, action="send", text="hi", session_override="sess-9").run()

    assert "resolve sess-9" in client.calls
    assert client.commands == [("user_message", "hi")]
    assert store.load("sess-9") is None


def test_unknown_session_override_is_actionable(tmp_path: Path) -> None:
    """A miss on the resolution route mentions the session id and the login."""
    store = SessionStateStore(tmp_path / "state")
    client = _FakeClient()

    with pytest.raises(typer.BadParameter, match="sess-9"):
        _command_driver(client, store, action="send", text="hi", session_override="sess-9").run()


# -- DetachedCommandDriver: attach -------------------------------------------------------------


class _InstantDetachReader:
    """A reader whose user immediately detaches (never ends)."""

    def __init__(self, *_args: object) -> None:
        self.detach = threading.Event()
        self.detach.set()

    def start(self) -> None:
        pass


def test_attach_detaches_without_ending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaving an attachment persists the cursor and leaves the session running."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.pages = [
        _page([], 0, "running"),
        _page([_event(1, "assistant_message", text="hello")], 1, "running"),
    ]
    monkeypatch.setattr(mod, "AttachedCommandReader", _InstantDetachReader)

    _command_driver(client, store, action="attach").run()

    assert client.commands == []
    assert client.end_calls == []
    state = store.load("sess-1")
    assert state is not None
    assert state.cursor == 1
    assert store.current_session_id() == "sess-1"


def test_attach_to_terminal_session_is_actionable(tmp_path: Path) -> None:
    """Attaching to a finished session explains the --end reconciliation path."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.session_states = [_session("failed")]

    with pytest.raises(typer.BadParameter, match="--end"):
        _command_driver(client, store, action="attach").run()


def test_attach_finishes_cleanly_when_the_session_ends_remotely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session ended from the web while attached tears down local state."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.session_states = [_session("running"), _session("ended")]
    client.pages = [
        _page([], 0, "running"),
        _page([_event(1, "state", status="idle")], 1, "ended"),
    ]

    class _NeverDetachReader:
        def __init__(self, *_args: object) -> None:
            self.detach = threading.Event()

        def start(self) -> None:
            pass

    monkeypatch.setattr(mod, "AttachedCommandReader", _NeverDetachReader)

    _command_driver(client, store, action="attach").run()

    assert store.load("sess-1") is None
    assert store.current_session_id() is None


# -- DetachedCommandDriver: end ----------------------------------------------------------------


def test_end_syncs_final_workspace_and_clears_state(tmp_path: Path) -> None:
    """--end pushes local edits, ends, applies the final archive, acks, and cleans up."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(root)
    store = _store_with_session(tmp_path, workspace_root=root, base_archive=base.archive)
    client = _FakeClient()
    client.session_states = [
        _session(workspace_sync=True),
        _session("ended", workspace_sync=True),
    ]
    client.final_archive = _archive({"answer.txt": b"after"})
    client.pages = [
        _page([], 0, "running"),
        _page([], 0, "ended"),
    ]

    _command_driver(client, store, action="end").run()

    assert client.end_calls == [("agent-1", "sess-1")]
    assert (root / "answer.txt").read_text(encoding="utf-8") == "after"
    assert client.final_acked
    assert store.load("sess-1") is None
    assert store.current_session_id() is None


def test_end_on_already_terminal_session_reconciles_and_clears(tmp_path: Path) -> None:
    """--end on a session that died while detached still syncs and clears state."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(root)
    store = _store_with_session(tmp_path, workspace_root=root, base_archive=base.archive)
    client = _FakeClient()
    client.session_states = [
        _session("ended", workspace_sync=True),
        _session("ended", workspace_sync=True),
    ]
    client.final_archive = _archive({"answer.txt": b"after"})

    _command_driver(client, store, action="end").run()

    assert client.end_calls == []
    assert (root / "answer.txt").read_text(encoding="utf-8") == "after"
    assert client.final_acked
    assert store.load("sess-1") is None


def test_failed_session_end_exits_nonzero_after_cleanup(tmp_path: Path) -> None:
    """A failed session's end reports the error and exits 1, with state removed."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.session_states = [_session("failed"), _session("failed")]

    with pytest.raises(typer.Exit) as raised:
        _command_driver(client, store, action="end").run()

    assert raised.value.exit_code == 1
    assert store.load("sess-1") is None


def test_remote_patch_does_not_absorb_unpushed_local_edits(tmp_path: Path) -> None:
    """A local edit made while detached must still upload after a patch lands.

    Reproduces the preview bug: advancing the base by re-snapshotting the local
    directory swallowed unpushed local files, so they never reached the sandbox
    and the final sync could even delete them.
    """
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(tmp_path)
    patch = build_workspace_patch(base.archive, _archive({"answer.txt": b"during"}))
    assert patch is not None
    (tmp_path / "local-only.txt").write_text("unpushed", encoding="utf-8")
    client = _FakeClient()
    client.patches["patch-1"] = patch
    workspace = mod.LiveWorkspace(
        cast("mod.PlatformClient", client), "agent-1", "sess-1", tmp_path, base
    )

    workspace.apply_remote_patch("patch-1")

    assert "local-only.txt" not in workspace.synchronized.files
    assert workspace.push_local()
    assert len(client.uploaded_patches) == 1


def test_attach_uploads_local_edits_before_rebasing_on_remote_patches(tmp_path: Path) -> None:
    """Catch-up pushes local changes first so a pending patch cannot swallow them."""
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(root)
    store = _store_with_session(tmp_path, workspace_root=root, base_archive=base.archive)
    patch = build_workspace_patch(base.archive, _archive({"answer.txt": b"during"}))
    assert patch is not None
    (root / "local-only.txt").write_text("created while detached", encoding="utf-8")
    client = _FakeClient()
    client.session_states = [_session(workspace_sync=True)]
    client.patches["patch-1"] = patch
    client.pages = [
        _page([_event(1, "workspace_patch", revision="patch-1")], 1, "running"),
        _page([], 1, "running"),
        _page(
            [_event(2, "user_message", text="hi"), _event(3, "state", status="idle")],
            3,
            "running",
        ),
    ]

    _command_driver(client, store, action="send", text="hi").run()

    assert len(client.uploaded_patches) >= 1
    assert client.calls.index("upload_patch") < client.calls.index("patch:patch-1")
    assert (root / "answer.txt").read_text(encoding="utf-8") == "during"


def test_attached_reader_reports_a_failed_end_and_keeps_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed :end must be reported, not silently converted into a detach."""
    monkeypatch.setattr(mod.sys, "stdin", io.StringIO(":end\n:stop\n"))

    class _FailingEndClient(_FakeClient):
        def end_agent_session(self, agent_id: str, session_id: str) -> RemoteAgentSession:
            raise PlatformError("backend unavailable", status_code=503)

    client = _FailingEndClient()
    reader = mod.AttachedCommandReader(cast("mod.PlatformClient", client), "agent-1", "sess-1")

    reader.run()

    # The reader kept consuming input after the failed end (the :stop landed),
    # and only stdin EOF flipped the detach event.
    assert client.commands == [("interrupt", None)]
    assert reader.end_failed
    assert reader.detach.is_set()


def test_detached_start_state_failure_names_the_running_session(
    tmp_path: Path,
) -> None:
    """A state-write failure after creation must hand the user the session id."""

    class _FailingSaveStore(SessionStateStore):
        def save(
            self, state: DetachedSessionState, *, base_archive: bytes | None = None
        ) -> DetachedSessionState:
            raise mod.SessionStateError("disk full")

    store = _FailingSaveStore(tmp_path / "state")
    client = _FakeClient()
    driver = mod.DetachedStartDriver(
        client=cast("mod.PlatformClient", client),
        credentials=_credentials(),
        state_store=store,
        target_id="agent-1",
        name="Agent",
        jail_root=None,
        task=None,
    )

    with pytest.raises(typer.BadParameter, match="sess-1"):
        driver.run()


def test_end_reruns_cleanly_while_the_session_is_ending(tmp_path: Path) -> None:
    """--end on a session already winding down still reaches the final handoff."""
    store = _store_with_session(tmp_path)
    client = _FakeClient()
    client.session_states = [_session("ending"), _session("ended")]
    client.end_result = _session("ending")
    client.pages = [
        _page([], 0, "ending"),
        _page([], 0, "ended"),
    ]

    _command_driver(client, store, action="end").run()

    assert client.end_calls == [("agent-1", "sess-1")]
    assert store.load("sess-1") is None


def test_try_push_local_tolerates_a_workspace_that_is_not_ready(tmp_path: Path) -> None:
    """A booting or winding-down sandbox defers the push; real failures still raise."""
    (tmp_path / "answer.txt").write_text("before", encoding="utf-8")
    base = snapshot_workspace(tmp_path)
    (tmp_path / "answer.txt").write_text("local", encoding="utf-8")

    class _NotReadyClient(_FakeClient):
        def __init__(self, status_code: int) -> None:
            super().__init__()
            self.status_code = status_code

        def upload_agent_workspace_patch(
            self, agent_id: str, session_id: str, content: bytes
        ) -> WorkspacePatchResult:
            raise PlatformError("workspace is not running", status_code=self.status_code)

    for status_code in (409, 503):
        workspace = mod.LiveWorkspace(
            cast("mod.PlatformClient", _NotReadyClient(status_code)),
            "agent-1",
            "sess-1",
            tmp_path,
            base,
        )
        assert workspace.try_push_local() is False
        assert workspace.synchronized is base

    failing = mod.LiveWorkspace(
        cast("mod.PlatformClient", _NotReadyClient(500)), "agent-1", "sess-1", tmp_path, base
    )
    with pytest.raises(PlatformError):
        failing.try_push_local()


def test_contested_path_survives_push_rejection_catchup_and_finalize(tmp_path: Path) -> None:
    """The full disputed ordering, end to end: a both-sides-edited file is never lost.

    Composes the pieces the ordering debate touched: a detached checkpoint with
    X at base content B, a local edit to U while detached, and a pending hosted
    patch moving the same X from B to A. The send flow pushes first (rejected
    with a content conflict), catch-up applies the patch (local conflict, file
    kept), and the end flow's finalize preserves U locally, saves A in the
    recovery archive, and signals the conflict with exit code 2.
    """
    root = tmp_path / "work"
    root.mkdir()
    (root / "answer.txt").write_text("B", encoding="utf-8")
    base = snapshot_workspace(root)
    store = _store_with_session(tmp_path, workspace_root=root, base_archive=base.archive)
    patch = build_workspace_patch(base.archive, _archive({"answer.txt": b"A"}))
    assert patch is not None
    (root / "answer.txt").write_text("U", encoding="utf-8")

    send_client = _FakeClient()
    send_client.session_states = [_session(workspace_sync=True)]
    send_client.upload_result = WorkspacePatchResult(applied=[], conflicts=["answer.txt"])
    send_client.patches["patch-1"] = patch
    send_client.pages = [
        _page([_event(1, "workspace_patch", revision="patch-1")], 1, "running"),
        _page([], 1, "running"),
        _page(
            [_event(2, "user_message", text="hi"), _event(3, "state", status="idle")],
            3,
            "running",
        ),
    ]

    _command_driver(send_client, store, action="send", text="hi").run()

    # The rejected push ran before the patch download (the disputed ordering).
    assert send_client.calls.index("upload_patch") < send_client.calls.index("patch:patch-1")
    assert send_client.patch_acks == ["patch-1"]
    # The local edit is untouched, recorded as a conflict, and the base kept B.
    assert (root / "answer.txt").read_text(encoding="utf-8") == "U"
    state = store.load("sess-1")
    assert state is not None
    assert state.workspace is not None
    assert state.workspace.conflicts == ("answer.txt",)
    checkpoint = snapshot_from_archive(store.load_base_archive(state))
    assert checkpoint.files["answer.txt"] == base.files["answer.txt"]

    end_client = _FakeClient()
    end_client.session_states = [
        _session(workspace_sync=True),
        _session("ended", workspace_sync=True),
    ]
    end_client.upload_result = WorkspacePatchResult(applied=[], conflicts=["answer.txt"])
    end_client.final_archive = _archive({"answer.txt": b"A"})
    end_client.pages = [
        _page([], 3, "running"),
        _page([], 3, "ended"),
    ]

    with pytest.raises(typer.Exit) as raised:
        _command_driver(end_client, store, action="end").run()

    assert raised.value.exit_code == 2
    assert (root / "answer.txt").read_text(encoding="utf-8") == "U"
    recovery = root / ".wmh-conflicts" / "sess-1.tar.gz"
    assert recovery.is_file()
    with tarfile.open(recovery) as archive:
        member = archive.extractfile("answer.txt")
        assert member is not None
        assert member.read() == b"A"
    assert end_client.final_acked
    assert store.load("sess-1") is None


def test_partially_conflicted_push_advances_accepted_sibling_paths(tmp_path: Path) -> None:
    """Accepted siblings of a rejected path advance the base; only the conflict stays.

    Keeping the whole base stale would re-push the accepted paths later against
    a base the sandbox has moved past, manufacturing conflicts if they change
    again locally.
    """
    (tmp_path / "answer.txt").write_text("B", encoding="utf-8")
    (tmp_path / "other.txt").write_text("O", encoding="utf-8")
    base = snapshot_workspace(tmp_path)
    (tmp_path / "answer.txt").write_text("U", encoding="utf-8")
    (tmp_path / "other.txt").write_text("O2", encoding="utf-8")
    client = _FakeClient()
    client.upload_result = WorkspacePatchResult(applied=["other.txt"], conflicts=["answer.txt"])
    workspace = mod.LiveWorkspace(
        cast("mod.PlatformClient", client), "agent-1", "sess-1", tmp_path, base
    )

    changed = workspace.push_local()

    assert not changed
    assert workspace.conflicts == {"answer.txt"}
    disk = snapshot_workspace(tmp_path)
    assert workspace.synchronized.files["other.txt"] == disk.files["other.txt"]
    assert workspace.synchronized.files["answer.txt"] == base.files["answer.txt"]

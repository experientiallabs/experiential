"""Exercise privileged helper behavior using only temporary files and high local ports."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from exp.runtime.capture import system_helper as helper

DOMAINS = ("api.anthropic.com", "api.openai.com")


@pytest.fixture
def state(tmp_path: Path) -> helper.HostsState:
    """Create isolated state whose ownership is the unprivileged test user's."""
    hosts = tmp_path / "hosts"
    hosts.write_bytes(b"127.0.0.1 localhost\n::1 localhost\n# original")
    hosts.chmod(0o644)
    return helper.HostsState(helper.SystemPaths(hosts, tmp_path / "state", os.geteuid()))


@pytest.mark.parametrize(
    "domain",
    ["localhost", "127.0.0.1", "::1", "*.openai.com", "api.openai.com\nelse", "a.local", "A.com"],
)
def test_reject_unsafe_domains(domain: str) -> None:
    """Only explicit DNS names can become hosts aliases."""
    with pytest.raises(helper.CaptureSystemError, match="Invalid provider hostname"):
        helper.validate_domains((domain,))


@pytest.mark.parametrize("port", [0, 443, 1023, 65536])
def test_reject_unsafe_port(port: int) -> None:
    """The privileged listener cannot forward to itself or another privileged port."""
    with pytest.raises(helper.CaptureSystemError, match="between 1024 and 65535"):
        helper.validate_port(port)


def test_restore_preserves_unrelated_edits_and_original_newline(state: helper.HostsState) -> None:
    """Reset removes only the owned block, including its leading newline."""
    original = state.paths.hosts.read_bytes()
    with state.locked():
        journal = state.activate(DOMAINS)
        active = state.paths.hosts.read_bytes()
        assert active == original + journal.block()
        assert b"127.0.0.1\tapi.openai.com\n::1\tapi.openai.com\n" in active
        later = b"\n192.0.2.9 unrelated.example\n"
        state.paths.hosts.write_bytes(active + later)
        assert state.recover()
        assert state.paths.hosts.read_bytes() == original + later
        assert not state.recover()
        assert not (state.paths.state / "journal.json").exists()


def test_existing_alias_conflict_never_mutates_hosts(state: helper.HostsState) -> None:
    """Case-insensitive trailing-dot aliases count as an existing user override."""
    before = b"192.0.2.1 different.example API.OPENAI.COM. # user override\n"
    state.paths.hosts.write_bytes(before)
    with (
        state.locked(),
        pytest.raises(helper.CaptureSystemError, match="already has a hosts entry"),
    ):
        state.activate(DOMAINS)
    assert state.paths.hosts.read_bytes() == before
    assert not (state.paths.state / "journal.json").exists()


@pytest.mark.parametrize("edit", ["mapping", "markers", "duplicate"])
def test_modified_owned_block_is_preserved(state: helper.HostsState, edit: str) -> None:
    """Recovery refuses to erase edited or ambiguous ownership evidence."""
    with state.locked():
        journal = state.activate(DOMAINS)
        current = state.paths.hosts.read_bytes()
        if edit == "mapping":
            current = current.replace(b"127.0.0.1\tapi.openai.com", b"192.0.2.2\tapi.openai.com")
        elif edit == "markers":
            current = b"\n".join(
                line for line in current.split(b"\n") if not line.startswith(b"# EXPERIENTIAL")
            )
        else:
            current += journal.block()
        state.paths.hosts.write_bytes(current)
        with pytest.raises(helper.CaptureSystemError):
            state.recover()
        assert state.paths.hosts.read_bytes() == current
        assert (state.paths.state / "journal.json").exists()


def test_journal_before_hosts_is_recoverable(state: helper.HostsState) -> None:
    """A crash between durable intent and hosts replace needs no hosts rewrite."""
    before = state.paths.hosts.read_bytes()
    with state.locked():
        state._write_journal(helper.CaptureJournal("a" * 32, DOMAINS))
        assert not state.recover()
        assert state.paths.hosts.read_bytes() == before
        assert not (state.paths.state / "journal.json").exists()


def test_orphaned_marker_requires_manual_inspection(state: helper.HostsState) -> None:
    """Missing ownership state cannot authorize removal of marked bytes."""
    with state.locked():
        state.activate(DOMAINS)
        (state.paths.state / "journal.json").unlink()
        before = state.paths.hosts.read_bytes()
        with pytest.raises(helper.CaptureSystemError, match="without a recovery journal"):
            state.recover()
        assert state.paths.hosts.read_bytes() == before


def test_bad_journal_cannot_supply_paths_or_restore_bytes(state: helper.HostsState) -> None:
    """A malformed journal never grants arbitrary restoration authority."""
    with state.locked():
        path = state.paths.state / "journal.json"
        path.write_text(json.dumps({"hosts": "/etc/passwd", "restore": "malicious"}))
        path.chmod(0o600)
        before = state.paths.hosts.read_bytes()
        with pytest.raises(helper.CaptureSystemError, match="recovery state is invalid"):
            state.recover()
        assert state.paths.hosts.read_bytes() == before
        assert path.exists()


def test_global_lock_rejects_second_capture_or_reset(state: helper.HostsState) -> None:
    """A second helper cannot reset routing from underneath the active owner."""
    with state.locked(), pytest.raises(helper.CaptureSystemError, match="already running"):
        with helper.HostsState(state.paths).locked():
            pytest.fail("second helper obtained lock")


@pytest.mark.parametrize("target", ["hosts", "state", "journal", "lock"])
def test_symlinks_are_rejected(state: helper.HostsState, tmp_path: Path, target: str) -> None:
    """System file operations never follow a user-substituted symlink."""
    other = tmp_path / "other"
    other.write_bytes(b"untouched")
    if target == "hosts":
        state.paths.hosts.unlink()
        state.paths.hosts.symlink_to(other)
    elif target == "state":
        state.paths.state.symlink_to(tmp_path, target_is_directory=True)
    else:
        state.paths.state.mkdir(mode=0o700)
        (state.paths.state / ("journal.json" if target == "journal" else "lock")).symlink_to(other)
    with pytest.raises((helper.CaptureSystemError, OSError)):
        with state.locked():
            state.recover()
    assert other.read_bytes() == b"untouched"


def test_group_writable_hosts_rejected(state: helper.HostsState) -> None:
    """The helper requires a root-controlled hosts file in production."""
    state.paths.hosts.chmod(0o664)
    with state.locked(), pytest.raises(helper.CaptureSystemError, match="unsafe ownership"):
        state.activate(DOMAINS)


def test_concurrent_hosts_edit_aborts_without_overwrite(state: helper.HostsState) -> None:
    """A stale initial read cannot replace a newer hosts edit."""
    before = state.paths.hosts.read_bytes()
    after = before + b"\n192.0.2.1 later.example\n"
    state.paths.hosts.write_bytes(after)
    with pytest.raises(helper.CaptureSystemError, match="Hosts changed"):
        state._replace_hosts(before, b"wrong replacement")
    assert state.paths.hosts.read_bytes() == after


@pytest.mark.parametrize("change", ["metadata", "replacement"])
def test_last_version_check_detects_metadata_and_path_changes(
    state: helper.HostsState, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Changes while preparing the replacement abort without erasing external content."""
    original = state.paths.hosts.read_bytes()
    external = original + b"\n192.0.2.15 external.example\n"
    original_copy = helper._copy_hosts_metadata

    def concurrent_copy(source_fd: int, target_fd: int) -> None:
        """Apply an independent edit after metadata copy and before the final check."""
        original_copy(source_fd, target_fd)
        if change == "metadata":
            state.paths.hosts.chmod(0o600)
        else:
            replacement = state.paths.hosts.with_name("independent-edit")
            replacement.write_bytes(external)
            replacement.chmod(0o644)
            os.replace(replacement, state.paths.hosts)

    monkeypatch.setattr(helper, "_copy_hosts_metadata", concurrent_copy)
    with pytest.raises(helper.CaptureSystemError, match="Hosts changed"):
        state._replace_hosts(original, b"unwanted stale contents")
    if change == "metadata":
        assert state.paths.hosts.read_bytes() == original
        assert stat.S_IMODE(state.paths.hosts.stat().st_mode) == 0o600
    else:
        assert state.paths.hosts.read_bytes() == external


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple ACL and file flags are macOS metadata")
def test_activation_and_recovery_preserve_apple_metadata(state: helper.HostsState) -> None:
    """Retain ACLs, extended attributes, and harmless flags across both atomic replacements."""
    hosts = state.paths.hosts
    subprocess.run(
        [
            "/usr/bin/xattr",
            "-w",
            "com.experiential.capture-test",
            "managed-device-metadata",
            str(hosts),
        ],
        check=True,
    )
    subprocess.run(["/usr/bin/chflags", "nodump", str(hosts)], check=True)
    subprocess.run(["/bin/chmod", "+a", "everyone allow read", str(hosts)], check=True)

    def acl() -> list[bytes]:
        """Read only ACL records for the temporary test hosts file."""
        result = subprocess.run(["/bin/ls", "-le", str(hosts)], capture_output=True, check=True)
        return result.stdout.splitlines()[1:]

    def attribute() -> bytes:
        """Read the custom extended attribute from the temporary file only."""
        result = subprocess.run(
            ["/usr/bin/xattr", "-p", "com.experiential.capture-test", str(hosts)],
            capture_output=True,
            check=True,
        )
        return result.stdout.rstrip(b"\n")

    def flags() -> int:
        """Read macOS file flags without exposing platform-specific Python types to Linux."""
        result = subprocess.run(
            ["/usr/bin/stat", "-f", "%f", str(hosts)], capture_output=True, check=True
        )
        return int(result.stdout)

    original_acl = acl()
    assert original_acl
    before = hosts.read_bytes()
    with state.locked():
        state.activate(DOMAINS)
        assert attribute() == b"managed-device-metadata"
        assert flags() & stat.UF_NODUMP
        assert acl() == original_acl
        assert state.recover()
        assert hosts.read_bytes() == before
        assert attribute() == b"managed-device-metadata"
        assert flags() & stat.UF_NODUMP
        assert acl() == original_acl


@pytest.mark.skipif(sys.platform != "darwin", reason="Immutable file flags are macOS metadata")
@pytest.mark.parametrize("enabled, disabled", [("uchg", "nouchg"), ("uappnd", "nouappnd")])
def test_immutable_and_append_only_hosts_rejected_before_transaction(
    state: helper.HostsState, enabled: str, disabled: str
) -> None:
    """Unsupported flags cannot create an undeletable temporary root-owned snapshot."""
    hosts = state.paths.hosts
    original = hosts.read_bytes()
    subprocess.run(["/usr/bin/chflags", enabled, str(hosts)], check=True)
    try:
        with state.locked(), pytest.raises(helper.CaptureSystemError, match="immutable or append"):
            state.activate(DOMAINS)
        assert hosts.read_bytes() == original
        assert not (state.paths.state / "journal.json").exists()
        assert not list(hosts.parent.glob(".hosts.exp-capture-*"))
    finally:
        subprocess.run(["/usr/bin/chflags", disabled, str(hosts)], check=True)


def test_helper_cli_has_no_filesystem_override() -> None:
    """Even a privileged caller cannot inject arbitrary paths through helper flags."""
    with pytest.raises(SystemExit) as raised:
        helper.main(["reset", "--hosts", "/tmp/arbitrary"])
    assert raised.value.code == 2


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Serve streaming bytes for relay tests, including the empty readiness connection."""
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


def test_relay_streams_both_ip_families_and_closes_connections() -> None:
    """Opaque streaming bytes survive both loopbacks without response buffering."""

    async def exercise() -> None:
        """Use ephemeral listeners, never the system HTTPS port."""
        server = await asyncio.start_server(_echo, "127.0.0.1", 0)
        relay = helper.LoopbackRelay(server.sockets[0].getsockname()[1], listen_port=0)
        clients: list[asyncio.StreamWriter] = []
        try:
            await relay.start()
            for listener in relay.servers:
                address = listener.sockets[0].getsockname()
                reader, writer = await asyncio.open_connection(address[0], address[1])
                clients.append(writer)
                payload = b"SSE/WebSocket/TLS opaque bytes\x00\xff" * 4096
                writer.write(payload)
                await writer.drain()
                assert await asyncio.wait_for(reader.readexactly(len(payload)), 3) == payload
            assert len(relay.connections) == 2
            await relay.close()
            assert not relay.connections
        finally:
            await relay.close()
            for writer in clients:
                writer.close()
                await writer.wait_closed()
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["missing_proxy", "occupied_listener", "refresh"])
def test_setup_failure_restores_hosts(state: helper.HostsState, failure: str) -> None:
    """Listener and cache failures cannot strand newly installed mappings."""

    async def exercise() -> None:
        """Induce each setup failure through test-owned listeners and callbacks."""
        before = state.paths.hosts.read_bytes()
        server = await asyncio.start_server(_echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        relay = helper.LoopbackRelay(
            port, listen_port=port if failure == "occupied_listener" else 0
        )
        if failure == "missing_proxy":
            server.close()
            await server.wait_closed()
        read_fd, write_fd = os.pipe()

        def refresh() -> None:
            """Simulate resolver cache failure after a successful hosts mutation."""
            if failure == "refresh":
                raise OSError("synthetic cache failure")

        def ready(_journal: helper.CaptureJournal) -> None:
            """No failure scenario may report active capture."""
            pytest.fail("failed setup reported ready")

        try:
            with pytest.raises((helper.CaptureSystemError, OSError)):
                await helper.serve_capture(state, relay, DOMAINS, read_fd, ready, refresh)
            assert state.paths.hosts.read_bytes() == before
            assert not relay.servers
        finally:
            os.close(read_fd)
            os.close(write_fd)
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())


@pytest.mark.parametrize("termination", ["eof", "timeout", "malformed"])
def test_owner_loss_restores_hosts(state: helper.HostsState, termination: str) -> None:
    """EOF and lease expiry restore routing without any action from the dead owner."""

    async def exercise() -> None:
        """Run the real lifecycle with isolated state and a short injected lease."""
        before = state.paths.hosts.read_bytes()
        server = await asyncio.start_server(_echo, "127.0.0.1", 0)
        relay = helper.LoopbackRelay(server.sockets[0].getsockname()[1], listen_port=0)
        read_fd, write_fd = os.pipe()
        refreshed: list[bytes] = []

        def refresh() -> None:
            """Record refresh order without invoking any macOS system operation."""
            refreshed.append(state.paths.hosts.read_bytes())

        def ready(journal: helper.CaptureJournal) -> None:
            """Terminate the owner only after both listener and mappings are active."""
            assert len(relay.servers) == 2
            assert journal.block() in state.paths.hosts.read_bytes()
            if termination == "eof":
                os.close(write_fd)
            elif termination == "malformed":
                os.write(write_fd, b"not a heartbeat\n")

        try:
            await asyncio.wait_for(
                helper.serve_capture(state, relay, DOMAINS, read_fd, ready, refresh, 0.05), 3
            )
            assert state.paths.hosts.read_bytes() == before
            assert len(refreshed) == 2
            assert refreshed[-1] == before
            assert not relay.servers
        finally:
            os.close(read_fd)
            if termination != "eof":
                os.close(write_fd)
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())


def test_heartbeat_renews_lease_but_partial_messages_do_not() -> None:
    """A wedged supervisor cannot keep capture active using incomplete control input."""

    async def exercise() -> None:
        """Keep the lease alive several intervals before sending an incomplete message."""
        read_fd, write_fd = os.pipe()
        stop = asyncio.Event()
        task = asyncio.create_task(helper.monitor_control(read_fd, stop, 0.1))
        try:
            for _ in range(4):
                os.write(write_fd, b"PING\n")
                await asyncio.sleep(0.04)
                assert not stop.is_set()
            os.write(write_fd, b"PIN")
            await asyncio.wait_for(stop.wait(), 1)
            await task
        finally:
            task.cancel()
            os.close(read_fd)
            os.close(write_fd)

    asyncio.run(exercise())


@pytest.fixture
def crashed_helper(state: helper.HostsState) -> Iterator[subprocess.Popen[bytes]]:
    """Run a child that owns a fake hosts transaction until externally terminated."""
    script = """
import os, sys, time
from pathlib import Path
from exp.runtime.capture.system_helper import HostsState, SystemPaths
state = HostsState(SystemPaths(Path(sys.argv[1]), Path(sys.argv[2]), os.geteuid()))
with state.locked():
    state.activate(("api.anthropic.com", "api.openai.com"))
    sys.stdout.write("ready\\n")
    sys.stdout.flush()
    time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(state.paths.hosts), str(state.paths.state)],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == b"ready\n"
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


def test_killed_helper_leaves_offline_recoverable_journal(
    state: helper.HostsState, crashed_helper: subprocess.Popen[bytes]
) -> None:
    """SIGKILL bypasses cleanup but releases the lock for a separate offline reset."""
    os.kill(crashed_helper.pid, signal.SIGKILL)
    crashed_helper.wait(timeout=5)
    original = b"127.0.0.1 localhost\n::1 localhost\n# original"
    journal = helper.CaptureJournal.decode((state.paths.state / "journal.json").read_bytes())
    assert state.paths.hosts.read_bytes() == original + journal.block()
    with helper.HostsState(state.paths).locked():
        assert helper.HostsState(state.paths).recover()
    assert state.paths.hosts.read_bytes() == original


@pytest.mark.skipif(sys.platform != "darwin", reason="Exercise Apple's standalone Python 3.9")
def test_apple_python_runs_relay_and_recovery_without_package_imports(
    state: helper.HostsState,
) -> None:
    """Run the real helper under Apple's isolated interpreter with synthetic paths only."""
    script = """
import asyncio, importlib.util, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("capture_helper", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)

async def echo(reader, writer):
    '''Echo streaming bytes under Apple's interpreter.'''
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()

async def exercise():
    '''Keep every test mutation inside injected temporary paths.'''
    paths = helper.SystemPaths(Path(sys.argv[2]), Path(sys.argv[3]), os.geteuid())
    state = helper.HostsState(paths)
    before = state.paths.hosts.read_bytes()
    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    relay = helper.LoopbackRelay(server.sockets[0].getsockname()[1], listen_port=0)
    read_fd, write_fd = os.pipe()
    started = asyncio.Event()
    serving = asyncio.create_task(helper.serve_capture(
        state, relay, ("api.openai.com",), read_fd, lambda _: started.set(), lambda: None, 0.5
    ))
    try:
        await asyncio.wait_for(started.wait(), 3)
        for listener in relay.servers:
            address = listener.sockets[0].getsockname()
            reader, writer = await asyncio.open_connection(address[0], address[1])
            payload = bytes(range(256)) * 1024
            writer.write(payload)
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(len(payload)), 3) == payload
            writer.close()
            await writer.wait_closed()
        # Leave the control pipe open: only heartbeat expiry may trigger cleanup.
        await asyncio.wait_for(serving, 3)
        assert state.paths.hosts.read_bytes() == before
        assert not relay.servers
    finally:
        os.close(write_fd)
        os.close(read_fd)
        server.close()
        await server.wait_closed()

asyncio.run(exercise())
"""
    subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            script,
            str(Path(helper.__file__)),
            str(state.paths.hosts),
            str(state.paths.state),
        ],
        check=True,
        timeout=10,
    )

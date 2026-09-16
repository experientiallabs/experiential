"""Isolated, standard-library-only macOS hosts and loopback relay helper.

The executable accepts no filesystem paths. It owns one marked hosts block and a
root-owned recovery journal, never a snapshot of somebody else's hosts file.
SIGINT/SIGTERM, control-pipe EOF, and an expired heartbeat lease request cleanup.
Power loss or killing this helper itself requires a later ``reset`` invocation.
Production evaluates an in-memory source snapshot using Apple's root-controlled
Python 3.9 or newer with ``-I -S``; it does not reopen this source file as root.
Hosts transactions use advisory locks and checked atomic replacements. Other
privileged hosts editors must coordinate with this helper; macOS does not offer
a compare-and-swap rename that excludes a noncooperating root writer.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":  # The public CLI can report macOS-only support on Windows.
    fcntl = None
else:
    import fcntl

LOGGER = logging.getLogger(__name__)
HEARTBEAT_TIMEOUT = 15.0
_MARKER = "# EXPERIENTIAL CAPTURE "
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_MAX_HOSTS_BYTES = 4 * 1024 * 1024
_MAX_JOURNAL_BYTES = 16384


def _file_version(info: os.stat_result) -> tuple[int, ...]:
    """Include metadata changes in the checked version of an open hosts inode."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_size,
        info.st_mode,
        info.st_uid,
        info.st_gid,
    )


def _copy_hosts_metadata(source_fd: int, target_fd: int) -> None:
    """Preserve macOS ACLs, extended attributes, flags, ownership, and permissions."""
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        copy = library.fcopyfile
        copy.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        copy.restype = ctypes.c_int
        # copyfile.h: COPYFILE_METADATA = COPYFILE_ACL | COPYFILE_STAT | COPYFILE_XATTR.
        if copy(source_fd, target_fd, None, 0b111) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    else:
        # The elevated CLI rejects non-macOS systems; portable attributes support Linux tests.
        info = os.fstat(source_fd)
        os.fchmod(target_fd, stat.S_IMODE(info.st_mode))
        if os.geteuid() == 0:
            os.fchown(target_fd, info.st_uid, info.st_gid)


class CaptureSystemError(RuntimeError):
    """A failed precondition or a recovery operation needing user attention."""


def validate_domains(domains: tuple[str, ...]) -> tuple[str, ...]:
    """Validate literal DNS names, rejecting addresses, wildcards, and local aliases."""
    if not domains or len(domains) > 32:
        raise CaptureSystemError("Capture needs between 1 and 32 explicit provider hostnames.")
    normalized = tuple(sorted(set(domains)))
    for domain in normalized:
        labels = domain.split(".")
        if (
            len(domain) > 253
            or len(labels) < 2
            or any(not _LABEL.fullmatch(label) for label in labels)
            or labels[-1] in {"localhost", "local", "internal", "test", "invalid"}
            or labels[-1].isdigit()
        ):
            raise CaptureSystemError(
                f"Invalid provider hostname {domain!r}; use a literal DNS name."
            )
    return normalized


def validate_port(port: int) -> int:
    """Require an unprivileged loopback target distinct from the capture listener."""
    if not 1024 <= port <= 65535:
        raise CaptureSystemError("The local capture proxy port must be between 1024 and 65535.")
    return port


@dataclass(frozen=True)
class SystemPaths:
    """Fixed production paths; alternate paths are injectable only through Python tests."""

    hosts: Path = Path("/private/etc/hosts")
    state: Path = Path("/private/var/db/experiential-capture")
    owner_uid: int = 0


@dataclass(frozen=True)
class CaptureJournal:
    """Validated facts sufficient to remove exactly one owned hosts block."""

    session: str
    domains: tuple[str, ...]

    def block(self) -> bytes:
        """Render the only bytes this session may add or remove from hosts."""
        rows = [f"\n{_MARKER}BEGIN {self.session}\n"]
        for domain in self.domains:
            rows.extend((f"127.0.0.1\t{domain}\n", f"::1\t{domain}\n"))
        rows.append(f"{_MARKER}END {self.session}\n")
        return "".join(rows).encode("ascii")

    def encoded(self) -> bytes:
        """Serialize only validated identifiers, never file paths or restore content."""
        return json.dumps({"version": 1, "session": self.session, "domains": self.domains}).encode()

    @classmethod
    def decode(cls, content: bytes) -> CaptureJournal:
        """Reject malformed state rather than guessing which system bytes to remove."""
        try:
            value = json.loads(content)
            if (
                not isinstance(value, dict)
                or set(value) != {"version", "session", "domains"}
                or type(value["version"]) is not int
                or value["version"] != 1
                or not isinstance(value["session"], str)
                or not re.fullmatch(r"[0-9a-f]{32}", value["session"])
                or not isinstance(value["domains"], list)
                or not all(isinstance(item, str) for item in value["domains"])
            ):
                raise ValueError("invalid journal fields")
            domains = validate_domains(tuple(value["domains"]))
            if list(domains) != value["domains"]:
                raise ValueError("noncanonical domain list")
            return cls(session=value["session"], domains=domains)
        except (ValueError, TypeError, UnicodeError) as exc:
            raise CaptureSystemError(
                "Capture recovery state is invalid. Leave hosts unchanged and inspect the "
                "root-owned capture journal."
            ) from exc


class HostsState:
    """Own a global lock and transaction journal while preserving unrelated hosts edits."""

    def __init__(self, paths: SystemPaths | None = None) -> None:
        """Store the filesystem boundary; production callers use the fixed paths."""
        self.paths = paths if paths is not None else SystemPaths()

    def _check_file(self, fd: int, *, private: bool) -> os.stat_result:
        """Require a regular, single-link, owner-controlled file before reading or writing."""
        info = os.fstat(fd)
        disallowed = 0o077 if private else 0o022
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != self.paths.owner_uid
            or info.st_nlink != 1
            or info.st_mode & disallowed
        ):
            raise CaptureSystemError("Capture system files have unsafe ownership or permissions.")
        if sys.platform == "darwin" and info.st_flags & (
            stat.UF_IMMUTABLE | stat.SF_IMMUTABLE | stat.UF_APPEND | stat.SF_APPEND
        ):
            raise CaptureSystemError("Capture system files must not be immutable or append-only.")
        return info

    def _ensure_state(self) -> None:
        """Create or validate a private directory without following a replacement symlink."""
        try:
            self.paths.state.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = self.paths.state.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != self.paths.owner_uid
            or info.st_mode & 0o077
        ):
            raise CaptureSystemError("Capture state directory must be private and owned by root.")

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold one nonblocking global lock for the entire capture or reset operation."""
        if fcntl is None:
            raise CaptureSystemError("DNS capture currently supports macOS only.")
        self._ensure_state()
        fd = os.open(self.paths.state / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            self._check_file(fd, private=True)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CaptureSystemError(
                    "Capture is already running. Stop it with Ctrl+C, then retry capture reset."
                ) from exc
            yield
        finally:
            os.close(fd)

    def _read(self, path: Path, *, private: bool, limit: int) -> bytes:
        """Read bounded content through a checked, non-symlink file descriptor."""
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = self._check_file(fd, private=private)
            if info.st_size > limit:
                raise CaptureSystemError("Capture system file exceeds the supported size limit.")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                content = stream.read(limit + 1)
            if len(content) > limit:
                raise CaptureSystemError("Capture system file exceeds the supported size limit.")
            return content
        finally:
            os.close(fd)

    def _sync_directory(self, path: Path) -> None:
        """Flush directory metadata after a durable journal or hosts transaction."""
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _write_journal(self, journal: CaptureJournal) -> None:
        """Persist intent before the hosts mutation so interrupted setup remains recoverable."""
        path = self.paths.state / "journal.json"
        if path.exists() or path.is_symlink():
            raise CaptureSystemError("Capture recovery state already exists. Run capture reset.")
        temporary = self.paths.state / f".journal-{uuid.uuid4().hex}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(journal.encoded())
                stream.flush()
                os.fsync(fd)
            os.replace(temporary, path)
        finally:
            os.close(fd)
            with suppress(FileNotFoundError):
                temporary.unlink()
        self._sync_directory(self.paths.state)

    def _replace_hosts(self, before: bytes, after: bytes) -> None:
        """Atomically replace a checked hosts file, aborting on detected concurrent edits."""
        if fcntl is None:
            raise CaptureSystemError("DNS capture currently supports macOS only.")
        path = self.paths.hosts
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        temporary = path.with_name(f".hosts.exp-capture-{uuid.uuid4().hex}")
        temporary_fd = -1
        try:
            info = self._check_file(fd, private=False)
            fcntl.flock(fd, fcntl.LOCK_EX)
            if self._read(path, private=False, limit=_MAX_HOSTS_BYTES) != before:
                raise CaptureSystemError("Hosts changed during capture setup. Retry the operation.")
            temporary_fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(temporary_fd, "wb", closefd=False) as stream:
                stream.write(after)
                stream.flush()
                _copy_hosts_metadata(fd, temporary_fd)
                os.fsync(temporary_fd)
            latest = path.lstat()
            if (
                _file_version(latest) != _file_version(info)
                or _file_version(os.fstat(fd)) != _file_version(info)
                or self._read(path, private=False, limit=_MAX_HOSTS_BYTES) != before
            ):
                raise CaptureSystemError("Hosts changed during capture setup. Retry the operation.")
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        finally:
            if temporary_fd >= 0:
                os.close(temporary_fd)
            with suppress(FileNotFoundError):
                temporary.unlink()
            os.close(fd)

    def recover(self) -> bool:
        """Remove only the exact journaled block, retaining state when someone edited it."""
        path = self.paths.state / "journal.json"
        try:
            content = self._read(path, private=True, limit=_MAX_JOURNAL_BYTES)
        except FileNotFoundError:
            hosts = self._read(self.paths.hosts, private=False, limit=_MAX_HOSTS_BYTES)
            if _MARKER.encode() in hosts:
                raise CaptureSystemError(
                    "Hosts contains capture markers without a recovery journal. "
                    "Inspect those entries before retrying."
                ) from None
            return False
        journal = CaptureJournal.decode(content)
        hosts = self._read(self.paths.hosts, private=False, limit=_MAX_HOSTS_BYTES)
        block = journal.block()
        if hosts.count(block) == 1:
            remaining = hosts.replace(block, b"", 1)
            if _MARKER.encode() in remaining:
                raise CaptureSystemError(
                    "Conflicting capture markers in hosts; inspect them first."
                )
            self._replace_hosts(hosts, remaining)
            changed = True
        elif _MARKER.encode() in hosts:
            raise CaptureSystemError(
                "Capture's hosts entries were edited. Reset left the file unchanged; "
                "restore the marked block to match the journal, then retry."
            )
        else:
            for line in hosts.decode("utf-8", errors="replace").splitlines():
                aliases = line.split("#", 1)[0].split()[1:]
                if any(alias.lower().rstrip(".") in journal.domains for alias in aliases):
                    raise CaptureSystemError(
                        "Capture markers are missing but provider hosts entries remain. "
                        "Inspect those entries before retrying reset."
                    )
            changed = False
        path.unlink()
        self._sync_directory(self.paths.state)
        return changed

    def activate(self, domains: tuple[str, ...]) -> CaptureJournal:
        """Journal and append exact loopback mappings after stale state has been recovered."""
        domains = validate_domains(domains)
        before = self._read(self.paths.hosts, private=False, limit=_MAX_HOSTS_BYTES)
        if _MARKER.encode() in before:
            raise CaptureSystemError("Capture markers already exist. Run capture reset first.")
        for line in before.decode("utf-8", errors="replace").splitlines():
            aliases = line.split("#", 1)[0].split()[1:]
            if any(alias.lower().rstrip(".") in domains for alias in aliases):
                raise CaptureSystemError(
                    "A provider hostname already has a hosts entry. Remove the conflicting "
                    "entry yourself before enabling capture."
                )
        journal = CaptureJournal(session=uuid.uuid4().hex, domains=domains)
        self._write_journal(journal)
        self._replace_hosts(before, before + journal.block())
        return journal


def refresh_dns() -> None:
    """Refresh the system resolver after changing hosts, without touching app connections."""
    for command in (
        ["/usr/bin/dscacheutil", "-flushcache"],
        ["/usr/bin/killall", "-HUP", "mDNSResponder"],
    ):
        subprocess.run(command, check=True, timeout=5, stdout=subprocess.DEVNULL)


class LoopbackRelay:
    """Forward opaque TCP bytes on IPv4 and IPv6 loopback without interpreting credentials."""

    def __init__(self, target_port: int, listen_port: int = 443) -> None:
        """Set one fixed loopback destination and a production HTTPS listening port."""
        self.target_port = validate_port(target_port)
        self.listen_port = listen_port
        self.servers: list[asyncio.Server] = []
        self.connections: set[asyncio.Task[None]] = set()

    async def check_target(self) -> None:
        """Require the unprivileged proxy listener before installing hosts entries."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.target_port), timeout=2
            )
        except (OSError, asyncio.TimeoutError) as exc:  # noqa: UP041 - Apple Python 3.9 compatibility.
            raise CaptureSystemError(
                "The local capture proxy is not listening. Start the proxy before changing hosts."
            ) from exc
        writer.close()
        await writer.wait_closed()

    async def start(self) -> None:
        """Bind both loopback families before allowing any hosts mutation."""
        await self.check_target()
        try:
            for address, family in (("127.0.0.1", socket.AF_INET), ("::1", socket.AF_INET6)):
                server = await asyncio.start_server(
                    self._accept, address, self.listen_port, family=family
                )
                self.servers.append(server)
        except OSError as exc:
            await self.close()
            raise CaptureSystemError(
                "Capture cannot bind IPv4 and IPv6 loopback port 443. "
                "Stop the conflicting listener and retry."
            ) from exc

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Track relays so shutdown closes every accepted connection."""
        task = asyncio.create_task(self._relay(reader, writer))
        self.connections.add(task)
        task.add_done_callback(self.connections.discard)

    async def _copy(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Copy bounded chunks with transport backpressure and propagate half-close."""
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
            await writer.drain()

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Bridge one client without logging, parsing, storing, or replaying its traffic."""
        upstream: asyncio.StreamWriter | None = None
        copies: list[asyncio.Task[None]] = []
        try:
            upstream_reader, upstream = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.target_port), timeout=2
            )
            copies = [
                asyncio.create_task(self._copy(reader, upstream)),
                asyncio.create_task(self._copy(upstream_reader, writer)),
            ]
            await asyncio.gather(*copies)
        except (OSError, asyncio.TimeoutError):  # noqa: UP041 - Apple Python 3.9 compatibility.
            LOGGER.debug("Capture relay connection ended.")
        finally:
            for task in copies:
                task.cancel()
            if copies:
                await asyncio.gather(*copies, return_exceptions=True)
            writer.close()
            if upstream is not None:
                upstream.close()
            with suppress(OSError):
                await writer.wait_closed()
            if upstream is not None:
                with suppress(OSError):
                    await upstream.wait_closed()

    async def close(self) -> None:
        """Close listeners and cancel active relays after routing has been restored."""
        for server in self.servers:
            server.close()
        for connection in self.connections:
            connection.cancel()
        if self.connections:
            await asyncio.gather(*self.connections, return_exceptions=True)
        for server in self.servers:
            await server.wait_closed()
        self.servers.clear()


async def monitor_control(fd: int, stop: asyncio.Event, timeout: float = HEARTBEAT_TIMEOUT) -> None:
    """Expire capture on pipe closure, malformed control input, or a missed heartbeat."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

    def receive() -> None:
        """Drain small control messages; overflow is treated as an expired owner."""
        try:
            data = os.read(fd, 4096)
        except OSError:
            data = b""
        if not data or queue.full():
            stop.set()
            loop.remove_reader(fd)
        else:
            queue.put_nowait(data)

    loop.add_reader(fd, receive)
    pending = b""
    deadline = time.monotonic() + timeout
    try:
        while not stop.is_set():
            try:
                data = await asyncio.wait_for(queue.get(), max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError:  # noqa: UP041 - Apple Python 3.9 compatibility.
                stop.set()
                break
            pending += data
            if len(pending) > 4096:
                stop.set()
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line == b"PING":
                    deadline = time.monotonic() + timeout
                else:
                    stop.set()
    finally:
        loop.remove_reader(fd)


async def serve_capture(
    state: HostsState,
    relay: LoopbackRelay,
    domains: tuple[str, ...],
    control_fd: int,
    ready: Callable[[CaptureJournal], None],
    refresh: Callable[[], None] = refresh_dns,
    heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
) -> None:
    """Own routing until the controlling foreground process stops or loses its lease."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, stop.set)
    try:
        with state.locked():
            if state.recover():
                refresh()
            await relay.start()
            monitor: asyncio.Task[None] | None = None
            try:
                journal = state.activate(domains)
                refresh()
                monitor = asyncio.create_task(monitor_control(control_fd, stop, heartbeat_timeout))
                ready(journal)
                await stop.wait()
            finally:
                if monitor is not None:
                    monitor.cancel()
                    with suppress(asyncio.CancelledError):
                        await monitor
                try:
                    if state.recover():
                        refresh()
                finally:
                    await relay.close()
    finally:
        for sig, previous in previous_handlers.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, previous)


def _emit(event: str, *, detail: str = "", session: str = "") -> None:
    """Write the credential-free JSON control protocol to the parent process."""
    sys.stdout.write(json.dumps({"event": event, "detail": detail, "session": session}) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Execute only fixed-path serve/reset operations after platform and privilege checks."""
    parser = argparse.ArgumentParser(description="Experiential temporary hosts capture helper")
    commands = parser.add_subparsers(dest="action", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--domain", action="append", required=True)
    commands.add_parser("reset")
    args = parser.parse_args(argv)
    try:
        if sys.platform != "darwin" or os.geteuid() != 0:
            raise CaptureSystemError(
                "The capture system helper requires macOS administrator access."
            )
        if os.getpgrp() != os.getpid():
            os.setsid()
        state = HostsState()
        if args.action == "reset":
            with state.locked():
                state.recover()
                refresh_dns()
            _emit("reset")
        else:
            domains = validate_domains(tuple(args.domain))
            relay = LoopbackRelay(validate_port(args.port))

            def ready(journal: CaptureJournal) -> None:
                """Publish readiness only after listener and hosts activation both succeed."""
                _emit("ready", session=journal.session)

            asyncio.run(serve_capture(state, relay, domains, sys.stdin.fileno(), ready))
            _emit("stopped")
        return 0
    except (CaptureSystemError, OSError, subprocess.SubprocessError) as exc:
        _emit("error", detail=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

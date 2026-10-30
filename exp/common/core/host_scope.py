"""The single-host coordination boundary that every EXP lock and lease relies on.

EXP serializes writers with OS file locks (`filelock`), SQLite's own file locks, and leases whose
liveness is proven by probing a holder's kernel lock. All of those give mutual exclusion between
processes of ONE machine. Two machines sharing a project directory over a network filesystem get
no such guarantee: NFS, SMB and FUSE locks are advisory at best and silently absent at worst, so
two writers can both "hold" a lock and both report success.

This module makes that assumption explicit and enforced instead of silent:

- `require_host_scope` binds a directory to the machine that first coordinated writes in it. The
  binding is a small marker file, created exclusively and never rewritten, beside the locks it
  protects. Every guarded lock checks the marker in its own directory, so whichever directory two
  machines share, they meet at the same marker and the second machine is refused. The project's
  content database root is bound the same way on its first write.
- Directories on a known network filesystem are refused unless the operator declares a single
  writer with ``EXP_SINGLE_WRITER=1``. The declaration only waives the filesystem check; the
  machine binding still applies. Filesystem detection reads ``/proc/self/mountinfo`` and is
  Linux-only; on other platforms the machine binding is the whole guard.
- `current_host` names this machine and boot for lease records, so a lease is only ever taken
  over on the machine that wrote it (see `require_same_host`).

The machine identity is, in order: ``EXP_MACHINE_ID`` when set and non-empty, read as the raw
machine identifier on any platform; the platform's own identifier (``/etc/machine-id`` or
``/var/lib/dbus/machine-id`` on Linux, ``IOPlatformUUID`` on macOS, ``MachineGuid`` on
Windows); otherwise, on a Linux host without a machine-id such as a minimal container image, an
EXP-generated random identifier created once, exclusively, at ``~/.local/state/exp/machine-id``
and never rewritten. That home is the effective user's account home, never ``HOME`` or an XDG
variable, so an environment-selected profile is still the same machine. The generated
identifier is per account and lives in the container's own filesystem, so a recreated container
gets a new one and is refused by the markers the old one wrote; a container whose project
directory outlives it, two accounts sharing one such project, or any host whose identity must
be pinned, set ``EXP_MACHINE_ID`` to one fixed value for every run.

Machine and boot identifiers are hashed under an EXP-specific prefix before they are written:
systemd documents ``/etc/machine-id`` as confidential, and EXP only needs equality. A generated
identifier is hashed under its own kind, so it never equals a platform identifier's hash.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":
    import winreg
else:
    import pwd

MARKER_NAME = ".exp-host"
SINGLE_WRITER_ENV = "EXP_SINGLE_WRITER"
MACHINE_ID_ENV = "EXP_MACHINE_ID"
GENERATED_MACHINE_ID_NAME = "machine-id"
_PROTOCOL = "exp-host-scope-v1"
_LINUX_MACHINE_ID_PATHS = (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id"))
_LINUX_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_CREATION_WINDOW_S = 1.0
NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "davfs",
        "fuse.ceph",
        "fuse.gcsfuse",
        "fuse.glusterfs",
        "fuse.rclone",
        "fuse.s3fs",
        "fuse.sshfs",
        "glusterfs",
        "gpfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb",
        "smb2",
        "smb3",
        "smbfs",
    }
)
_REMEDY = (
    "Run every EXP command for this project on one machine, or execute agents on other "
    "machines through a remote EnvironmentRuntime while this machine keeps the project; "
    "EXP has no multi-host coordination backend yet."
)


class HostScopeError(RuntimeError):
    """A coordinated write would break EXP's single-host mutual-exclusion assumption."""


@dataclass(frozen=True)
class HostIdentity:
    """This machine and its current boot, as EXP records them.

    Attributes:
        machine: Hashed stable machine identifier; equal across reboots of one machine.
        boot: Hashed boot identifier, or ``None`` where the platform exposes none (Windows).
        hostname: Diagnostic name for messages; never used for equality.
    """

    machine: str
    boot: str | None
    hostname: str


def _digest(kind: str, value: str) -> str:
    """Hash one raw platform identifier so the raw value is never persisted."""
    return hashlib.sha256(f"{_PROTOCOL}:{kind}:{value.strip()}".encode()).hexdigest()[:32]


def _read_first(paths: tuple[Path, ...]) -> str | None:
    """Return the first non-empty identifier file among the standard locations."""
    for path in paths:
        try:
            value = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            continue
        if value:
            return value
    return None


def _darwin_value(command: tuple[str, ...], key: str | None) -> str:
    """Read one identifier from a macOS system tool."""
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    if key is None:
        return output.strip()
    for line in output.splitlines():
        if f'"{key}"' in line:
            return line.split("=", 1)[1].strip().strip('"')
    raise HostScopeError(f"cannot read {key} from {' '.join(command)}")


def _raw_identity() -> tuple[str | None, str | None]:
    """Read this platform's raw machine and boot identifiers.

    Returns:
        The stable machine identifier, or ``None`` on a Linux host without a machine-id, and the
        boot identifier, if the platform has one.

    Raises:
        HostScopeError: A macOS system tool reported no identifier.
    """
    if sys.platform == "win32":
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            machine, _ = winreg.QueryValueEx(key, "MachineGuid")
        return str(machine), None
    if sys.platform == "darwin":
        machine = _darwin_value(("ioreg", "-rd1", "-c", "IOPlatformExpertDevice"), "IOPlatformUUID")
        return machine, _darwin_value(("sysctl", "-n", "kern.bootsessionuuid"), None)
    return _read_first(_LINUX_MACHINE_ID_PATHS), _read_first((_LINUX_BOOT_ID_PATH,))


def _create_once(path: Path, payload: bytes, mode: int) -> bytes:
    """Create ``path`` exclusively with ``payload``, or read what another process created first.

    The payload is written in one call right after the exclusive creation, so a reader can only
    observe the file empty during that instant; the bounded re-read covers exactly that window.

    Returns:
        The file's contents, empty only if its creator never finished writing.
    """
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    deadline = time.monotonic() + _CREATION_WINDOW_S
    content = path.read_bytes()
    while not content and time.monotonic() < deadline:
        time.sleep(0.01)
        content = path.read_bytes()
    return content


def _account_home() -> Path:
    """Return the effective user's home from the account database, ignoring ``HOME``.

    Raises:
        HostScopeError: The effective user has no account entry with an absolute home.
    """
    try:
        home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except KeyError as exc:
        raise HostScopeError(
            f"this machine has no /etc/machine-id and user {os.geteuid()} has no account home "
            f"for EXP's own machine identity. Set {MACHINE_ID_ENV} to a fixed identifier for "
            "this machine, so EXP can keep project locks on one host."
        ) from exc
    if not home.is_absolute():
        raise HostScopeError(
            f"this machine has no /etc/machine-id and the account home {home} is not absolute. "
            f"Set {MACHINE_ID_ENV} to a fixed identifier for this machine, so EXP can keep "
            "project locks on one host."
        )
    return home


def generated_machine_id_path() -> Path:
    """Return where EXP keeps the identity it generates for a host without a machine-id."""
    return _account_home() / ".local" / "state" / "exp" / GENERATED_MACHINE_ID_NAME


def _generated_machine_id() -> str:
    """Return this user's EXP-generated machine identifier, creating it on first use.

    Raises:
        HostScopeError: The identifier file cannot be created or read, or is empty.
    """
    path = generated_machine_id_path()
    remedy = (
        f"Set {MACHINE_ID_ENV} to a fixed identifier for this machine, or make {path} "
        "creatable, so EXP can keep project locks on one host."
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        value = _create_once(path, uuid.uuid4().hex.encode("ascii"), 0o600).decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise HostScopeError(
            f"this machine has no /etc/machine-id, and EXP's own machine identity {path} cannot "
            f"be created or read ({exc}). {remedy}"
        ) from exc
    if not value.strip():
        raise HostScopeError(
            f"this machine has no /etc/machine-id, and EXP's own machine identity {path} is "
            f"empty; delete it to generate a new one. {remedy}"
        )
    return value


@functools.cache
def current_host() -> HostIdentity:
    """Return this machine's identity, read once per process.

    Returns:
        The hashed machine and boot identity with a diagnostic hostname.

    Raises:
        HostScopeError: The platform's identifiers are unreadable, or no machine identity can be
            read or generated.
    """
    try:
        machine, boot = _raw_identity()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HostScopeError(f"cannot identify this machine for project locking: {exc}") from exc
    override = os.environ.get(MACHINE_ID_ENV, "")
    if override.strip():
        machine_digest = _digest("machine", override)
    elif machine is not None:
        machine_digest = _digest("machine", machine)
    else:
        machine_digest = _digest("generated-machine", _generated_machine_id())
    return HostIdentity(
        machine=machine_digest,
        boot=None if boot is None else _digest("boot", boot),
        hostname=socket.gethostname(),
    )


def parse_mount_types(mountinfo: str) -> tuple[tuple[str, str], ...]:
    """Return ``(mount point, filesystem type)`` pairs from ``/proc/self/mountinfo`` text.

    Args:
        mountinfo: Contents in the kernel's mountinfo format, later mounts last.

    Returns:
        Each mount point with octal escapes decoded, paired with its filesystem type.
    """
    mounts = []
    for line in mountinfo.splitlines():
        fields, separator, rest = line.partition(" - ")
        parts = fields.split(" ")
        if not separator or len(parts) < 5 or not rest:
            continue
        point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), parts[4])
        mounts.append((point, rest.split(" ", 1)[0]))
    return tuple(mounts)


def filesystem_type(directory: Path) -> str | None:
    """Return the filesystem type holding ``directory``, or ``None`` off Linux.

    The innermost mount containing the resolved path wins; among equal mount points the latest
    mount does, because it shadows the earlier ones.

    Args:
        directory: An existing directory.

    Returns:
        The kernel filesystem type name, such as ``ext4`` or ``nfs4``, or ``None`` when the
        platform exposes no mount table.
    """
    try:
        text = _MOUNTINFO_PATH.read_text(encoding="utf-8", errors="surrogateescape")
    except FileNotFoundError:
        return None
    resolved = directory.resolve()
    best: tuple[int, str] | None = None
    for point, kind in parse_mount_types(text):
        mount = Path(point)
        if (resolved == mount or mount in resolved.parents) and (
            best is None or len(mount.parts) >= best[0]
        ):
            best = (len(mount.parts), kind)
    return None if best is None else best[1]


def _single_writer_declared() -> bool:
    """Read the explicit single-writer declaration, rejecting ambiguous values."""
    value = os.environ.get(SINGLE_WRITER_ENV, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise HostScopeError(f"{SINGLE_WRITER_ENV} must be 1 or unset, not {value!r}")


def _marker_payload(host: HostIdentity) -> bytes:
    """Serialize the immutable owning-host record for one directory."""
    return json.dumps(
        {"protocol": _PROTOCOL, "machine": host.machine, "hostname": host.hostname},
        sort_keys=True,
    ).encode()


def _claim_or_read(marker: Path, host: HostIdentity) -> dict[str, str]:
    """Create the marker exclusively, or read the one another process created first."""
    payload = _create_once(marker, _marker_payload(host), 0o644)
    try:
        record = json.loads(payload)
    except (OSError, ValueError) as exc:
        raise HostScopeError(
            f"the project host marker {marker} is unreadable ({exc}); if every EXP process "
            "for this project runs on this machine, delete the marker and retry"
        ) from exc
    if not isinstance(record, dict) or record.get("protocol") != _PROTOCOL:
        raise HostScopeError(
            f"the project host marker {marker} is not an {_PROTOCOL} record; if every EXP "
            "process for this project runs on this machine, delete the marker and retry"
        )
    return {str(key): str(value) for key, value in record.items()}


_verified: set[Path] = set()
_verified_lock = threading.Lock()


def require_host_scope(directory: Path) -> None:
    """Bind ``directory`` to this machine on first use, and refuse any other machine after.

    Call before taking a cross-process lock in ``directory`` or writing state whose consistency
    depends on such locks. Verified directories are remembered for the process: the marker is
    never rewritten, so a successful check cannot later become wrong.

    Args:
        directory: The existing directory that holds the lock files or database being
            coordinated.

    Raises:
        HostScopeError: Another machine owns the directory, the directory is on a network
            filesystem without a single-writer declaration, or this machine cannot be identified.
        OSError: The directory does not exist or the marker cannot be created.
    """
    key = Path(os.path.abspath(directory))
    if key in _verified:
        return
    with _verified_lock:
        if key in _verified:
            return
        kind = filesystem_type(key)
        if kind in NETWORK_FILESYSTEMS and not _single_writer_declared():
            raise HostScopeError(
                f"{key} is on a {kind} network filesystem, where EXP's file locks do not give "
                "mutual exclusion between machines. Move the project to a local disk, or, if "
                f"exactly one machine will ever write it, set {SINGLE_WRITER_ENV}=1. {_REMEDY}"
            )
        host = current_host()
        record = _claim_or_read(key / MARKER_NAME, host)
        if record.get("machine") != host.machine:
            raise HostScopeError(
                f"{key} is coordinated by another machine ({record.get('hostname', 'unknown')}),"
                f" and this machine ({host.hostname}) cannot share its locks: two hosts on one "
                f"project would silently break mutual exclusion. {_REMEDY} If the project was "
                f"moved here and the other machine no longer uses it, delete {key / MARKER_NAME}."
                f" If this is the same machine under a new identity (a container recreated "
                f"without /etc/machine-id), set {MACHINE_ID_ENV} to one fixed value for every "
                "run."
            )
        _verified.add(key)


def require_same_host(holder_machine: str, *, what: str) -> None:
    """Refuse to judge or take over a lease that another machine recorded.

    A lease's liveness is proven with this machine's process table and kernel locks, which say
    nothing about a process on another machine. Such a lease is never stale from here.

    Args:
        holder_machine: The hashed machine identity recorded by the lease holder.
        what: Noun naming the lease for the error message.

    Raises:
        HostScopeError: The holder ran on a different machine.
    """
    if holder_machine != current_host().machine:
        raise HostScopeError(
            f"{what} is held by a process on another machine; EXP can only verify and recover "
            f"leases taken on this machine. {_REMEDY}"
        )

"""Preflight checks and per-user ownership of the macOS local redirector."""

from __future__ import annotations

import os
import platform
import plistlib
import re
import stat
import subprocess
import sys
import tarfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.parsers.expat import ExpatError

from filelock import FileLock, Timeout

if sys.platform == "win32":
    pwd = None
else:
    import pwd

_APPLICATIONS = Path("/Applications")
_APP_NAME = "Mitmproxy Redirector.app"
_APP_IDENTIFIER = "org.mitmproxy.macos-redirector"
_EXTENSION_IDENTIFIER = f"{_APP_IDENTIFIER}.network-extension"
_EXTENSION_PATH = f"Contents/Library/SystemExtensions/{_EXTENSION_IDENTIFIER}.systemextension"
_APP_PLIST = f"{_APP_NAME}/Contents/Info.plist"
_EXTENSION_PLIST = f"{_APP_NAME}/{_EXTENSION_PATH}/Contents/Info.plist"
_ARCHIVE_MAX_BYTES = 32 * 1024 * 1024
_ARCHIVE_MAX_MEMBERS = 256
_REINSTALL = (
    "Capture's macOS redirector package is missing or invalid. "
    "Reinstall Experiential with Python 3.13 or newer to restore its mitmproxy-macos dependency."
)


@contextmanager
def capture_instance() -> Iterator[None]:
    """Hold one foreground Capture session per user, including startup and shutdown.

    This location uses the OS account's home, ignoring HOME, profile roots, and XDG
    settings: every session controls the same macOS redirector. The OS releases the
    lock when a process exits or crashes. Keeping the file preserves its inode for
    other waiting processes.

    Yields:
        None while this process owns the user's Capture session.

    Raises:
        RuntimeError: Another session is active or the lock path is unsafe or inaccessible.
    """
    try:
        path = _foreground_lock_path()
        lock = FileLock(path, timeout=0, mode=0o600)
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError(
            "Capture is already running for this macOS user. Stop that session before "
            "starting another one."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Capture could not acquire its foreground lock. Check access to "
            "~/Library/Application Support/exp/capture and rerun exp capture."
        ) from exc
    try:
        yield
    finally:
        lock.release()


def _foreground_lock_path() -> Path:
    """Create only user-owned directories and reject redirected or shared lock paths."""
    home = _account_home()
    for ancestor in reversed(home.parents):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            raise RuntimeError(f"Capture's home directory has an unsafe ancestor: {ancestor}")
    _require_owned_directory(home)
    directory = home
    for name in ("Library", "Application Support", "exp", "capture"):
        directory /= name
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _require_owned_directory(directory)
    # Older installations may have created this application-owned directory with 0755.
    # Normalize this narrow directory only, never the user's Library or home directory.
    directory.chmod(0o700)
    path = directory / "foreground.lock"
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return path
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
    ):
        raise RuntimeError(
            f"Capture's foreground lock is unsafe: {path}. Restore a regular file owned "
            "only by your user, then rerun exp capture."
        )
    return path


def _account_home() -> Path:
    """Resolve the effective OS account without trusting environment-selected profiles."""
    if pwd is None:
        raise RuntimeError("System Capture currently supports macOS only.")
    try:
        home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except (KeyError, OSError) as exc:
        raise RuntimeError(
            "Capture could not resolve your macOS account home. Check your account's "
            "home directory with your administrator, then rerun exp capture."
        ) from exc
    if not home.is_absolute():
        raise RuntimeError("Capture requires an absolute home directory for your macOS account.")
    return home


def _require_owned_directory(path: Path) -> None:
    """Reject symlinks, foreign ownership, and directories writable by other users."""
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise RuntimeError(
            f"Capture's foreground lock directory is unsafe: {path}. Use a directory "
            "owned by your user that other users cannot modify, then rerun exp capture."
        )


def require_local_backend(*, installation_is_current: Callable[[], bool]) -> None:
    """Validate prerequisites without installing or activating a Network Extension.

    The actual mitmproxy startup owns app installation and macOS authorization.
    Passing this check does not establish that the user has approved the extension.

    Args:
        installation_is_current: Native installer's read-only bundle identity check.

    Raises:
        RuntimeError: The platform, packaged app, or installation permissions prevent startup.
    """
    if sys.platform != "darwin":
        raise RuntimeError("System Capture currently supports macOS only.")
    if os.geteuid() == 0:
        raise RuntimeError("Run exp capture as your normal user, not with sudo.")
    archive = _packaged_archive()
    minimum = _minimum_macos(archive)
    current = _version(platform.mac_ver()[0])
    if current is None or current < minimum:
        required = ".".join(str(part) for part in minimum[:2])
        raise RuntimeError(f"Capture's local redirector requires macOS {required} or newer.")
    if os.environ.get("MITMPROXY_KEEP_REDIRECTOR") == "1":
        raise RuntimeError(
            "Unset MITMPROXY_KEEP_REDIRECTOR and rerun exp capture so mitmproxy can manage "
            "its packaged redirector."
        )
    _verify_packaged_app(archive)
    _require_install_access(installation_is_current)


def _packaged_archive() -> Path:
    """Find the official dependency's app archive without importing or starting its code."""
    try:
        package = distribution("mitmproxy-macos")
        archive = Path(str(package.locate_file(f"mitmproxy_macos/{_APP_NAME}.tar")))
        if not archive.is_file() or archive.stat().st_size > _ARCHIVE_MAX_BYTES:
            raise RuntimeError(_REINSTALL)
        return archive
    except (PackageNotFoundError, OSError) as exc:
        raise RuntimeError(_REINSTALL) from exc


def _version(value: str) -> tuple[int, int, int] | None:
    """Parse a numeric macOS release without assuming a marketing version name."""
    if re.fullmatch(r"[0-9]{1,4}(?:\.[0-9]{1,4}){0,2}", value) is None:
        return None
    parts = [int(part) for part in value.split(".")]
    parts.extend([0] * (3 - len(parts)))
    return parts[0], parts[1], parts[2]


def _minimum_macos(archive: Path) -> tuple[int, int, int]:
    """Read both bundle deployment targets in place, never extracting executable files."""
    minimum = (0, 0, 0)
    try:
        with tarfile.open(archive, "r:") as bundle:
            for name in (_APP_PLIST, _EXTENSION_PLIST):
                member = bundle.getmember(name)
                if not member.isfile() or not 0 < member.size <= 65536:
                    raise ValueError("invalid app metadata")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("missing app metadata")
                with source:
                    metadata = plistlib.loads(source.read(65537))
                version = (
                    metadata.get("LSMinimumSystemVersion") if isinstance(metadata, dict) else None
                )
                parsed = _version(version) if isinstance(version, str) else None
                if parsed is None:
                    raise ValueError("invalid app deployment target")
                minimum = max(minimum, parsed)
    except (
        OSError,
        KeyError,
        ValueError,
        tarfile.TarError,
        plistlib.InvalidFileException,
        ExpatError,
    ) as exc:
        raise RuntimeError(_REINSTALL) from exc
    return minimum


def _verify_packaged_app(archive: Path) -> None:
    """Verify a bounded private copy before mitmproxy can install or execute this archive.

    Validating the source bundle covers fresh installs before the native installer
    runs. The separate installed-bundle check covers reuse of identical app contents.
    """
    try:
        if archive.stat().st_size > _ARCHIVE_MAX_BYTES:
            raise ValueError("redirector archive is too large")
        with TemporaryDirectory(prefix="exp-capture-verify-") as temporary:
            destination = Path(temporary)
            _extract_verification_bundle(archive, destination)
            _verify_bundle_signature(destination / _APP_NAME)
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise RuntimeError(_REINSTALL) from exc


def _extract_verification_bundle(archive: Path, destination: Path) -> None:
    """Extract only bounded plain files and directories inside the single expected app."""
    names: set[str] = set()
    total_size = 0
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle:
            name = member.name.rstrip("/")
            parts = name.split("/")
            if (
                len(names) >= _ARCHIVE_MAX_MEMBERS
                or name in names
                or parts[0] != _APP_NAME
                or any(part in ("", ".", "..") for part in parts)
                or "\\" in name
                or not (member.isdir() or member.isreg())
                or member.issparse()
                or member.mode & 0o6022
                or (len(parts) == 1 and not member.isdir())
                or member.size < 0
            ):
                raise ValueError("unsafe redirector archive member")
            names.add(name)
            total_size += member.size
            if total_size > _ARCHIVE_MAX_BYTES:
                raise ValueError("redirector archive contents are too large")
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("missing redirector archive member")
            with source, target.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("truncated redirector archive member")
                    output.write(chunk)
                    remaining -= len(chunk)
            target.chmod(0o600 | (member.mode & 0o100))


def _verify_bundle_signature(app: Path) -> None:
    """Require intact app and extension signatures from mitmproxy's Apple Developer ID."""
    for bundle, identifier in (
        (app, _APP_IDENTIFIER),
        (app / _EXTENSION_PATH, _EXTENSION_IDENTIFIER),
    ):
        requirement = (
            "=anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists "
            "and certificate leaf[field.1.2.840.113635.100.6.1.13] exists "
            'and certificate leaf[subject.OU] = "S8XHQB96PW" '
            f'and identifier "{identifier}"'
        )
        try:
            result = subprocess.run(
                [
                    "/usr/bin/codesign",
                    "--verify",
                    "--strict",
                    "--deep",
                    "--all-architectures",
                    "--test-requirement",
                    requirement,
                    str(bundle),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Capture could not verify the Mitmproxy Redirector signature. "
                "Check that /usr/bin/codesign works, then rerun exp capture."
            ) from exc
        if result.returncode:
            raise RuntimeError(
                "Capture rejected the Mitmproxy Redirector app or extension signature. "
                "Reinstall Experiential; if the installed app is damaged, ask an administrator "
                "to remove /Applications/Mitmproxy Redirector.app before retrying."
            )


def _require_install_access(installation_is_current: Callable[[], bool]) -> None:
    """Use the native installer's content comparison before requesting replacement access."""
    if sys.platform != "darwin":
        raise RuntimeError("System Capture currently supports macOS only.")
    app = _APPLICATIONS / _APP_NAME
    executable = app / "Contents/MacOS/Mitmproxy Redirector"
    try:
        if installation_is_current():
            if executable.is_file() and os.access(executable, os.X_OK):
                _verify_bundle_signature(app)
                return
            raise RuntimeError(
                "The installed Mitmproxy Redirector app is incomplete. Ask an administrator "
                "to remove it from /Applications, then rerun exp capture to reinstall it."
            )
        if _writable_directory(_APPLICATIONS) and (not app.exists() or _writable_directory(app)):
            return
    except OSError as exc:
        raise RuntimeError("Capture could not inspect its local redirector installation.") from exc
    raise RuntimeError(
        "Capture needs permission to install or update /Applications/Mitmproxy Redirector.app. "
        "Ask your administrator to grant this user installation access, then rerun exp capture "
        "as your normal user."
    )


def _writable_directory(path: Path) -> bool:
    """Inspect access without creating files or requesting elevated privileges."""
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
